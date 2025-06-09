import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from copy import deepcopy
import json

from ..utils.tool import batchify
from .loss import softmax_entropy, mcc_loss, div_loss


class SUTASystem(object):

    SAMPLE_RATE = 16000

    def __init__(self, config) -> None:
        self.config = config
        self.history = {}
        self.adapt_count = 0

        # load processor and model
        raw_processor_no_lm = Wav2Vec2Processor.from_pretrained(config["model_name"], sampling_rate=SUTASystem.SAMPLE_RATE)
        self.raw_processor_no_lm = raw_processor_no_lm
        if config.get("use_lm", False) or config["model_name"] == "patrickvonplaten/wav2vec2-base-960h-4-gram":
            from transformers import Wav2Vec2ProcessorWithLM
            ngram_decoder = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-960h-4-gram").decoder
            self.processor = Wav2Vec2ProcessorWithLM(
                feature_extractor=raw_processor_no_lm.feature_extractor,
                tokenizer=raw_processor_no_lm.tokenizer,
                decoder=ngram_decoder
            )
        else:
            self.processor = raw_processor_no_lm
        
        if config["model_name"] == "facebook/data2vec-audio-base-960h":
            from transformers import Data2VecAudioForCTC
            self.model = Data2VecAudioForCTC.from_pretrained(config["model_name"], ctc_loss_reduction="mean")
        elif config["model_name"] == "facebook/hubert-large-ls960-ft":
            from transformers import HubertForCTC
            self.model = HubertForCTC.from_pretrained(config["model_name"], ctc_loss_reduction="mean")
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(config["model_name"], ctc_loss_reduction="mean")  # be careful that we need to use mean
        
        self.model.train()  # huggingface default loads with eval mode
        self.model.cuda()

        # set up for tent
        self.optimizer, self.scheduler = setup_optimizer(
            self.build_optimized_model(),
            config["opt"], config["lr"], scheduler=config["scheduler"]
        )

        f = open('vocab.json')
        self.vocab = json.load(f)

        self.snapshot("init")

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def build_optimized_model(self):
        self.model.requires_grad_(False)
        params, self.opt_param_names = self.collect_params()
        # print(param_names[:10])
        for p in params:
            p.requires_grad = True
        print("Optimizable: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        return params

    def _wav_to_model_input(self, wavs):
        # Due to wav2vec2-base special design, no attention mask is returned.
        # Wav2Vec2Processor's default argument for return_attention_mask will be False here.
        # However, it will be True in other speech models!
        inputs = self.processor(
            audio=wavs,
            sampling_rate=SUTASystem.SAMPLE_RATE,
            return_tensors="pt",
            padding="longest",
        )
        return inputs.to(device=self.model.device)
    
    def _text_to_model_input(self, texts):
        # target texts need to match wav2vec2's format to make sure correct tokenization
        texts_new = []
        for x in texts:
            x = x.upper()
            x_new = ""
            for s in x:
                if s in self.vocab or s == ' ':
                    x_new += s
            texts_new.append(x_new)

        labels = self.processor(
            text=texts_new,
            return_tensors="pt",
            padding="longest",
        )
        labels = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        return labels.to(device=self.model.device)

    def reset_adapt_counter(self):
        self.adapt_count = 0
    
    def l2_loss(self):
        l2_loss = 0.0
        orig_state_dict = self.history["init"][0]

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                l2_loss += torch.sum((param - orig_state_dict[name]) ** 2)
        return l2_loss

    def suta_adapt(self, wavs, record={}):
        """
        Single gradient step on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        the index of <pad> in vocab is 0
        Due to wav2vec2-base special design, attention mask is always none, so ctc input length is always the
        full length, model should learn to output id=0 on the padded part of wav.
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        # print(type(inputs))  # inputs belongs to a custom dict class defined in transformers, not tensor
        inputs = inputs.to(device=self.model.device)
        outputs = self.model(**inputs)
        predicted_ids = torch.argmax(outputs.logits, dim=-1)

        loss = 0
        if self.config["em_coef"] > 0:
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
            x = softmax_entropy(outputs.logits / self.config["temp"])
            if self.config["non_blank"]:
                x = x[non_blank]
            if len(x) > 0:
                e_loss = x.mean(0).mean()
            else:
                e_loss = torch.tensor(0, device=self.model.device)
                record["collapse"] = True
            loss += e_loss * self.config["em_coef"]
            record["e_loss"] = e_loss.item()
        
        if 1 - self.config["em_coef"] > 0: 
            c_loss = mcc_loss(outputs.logits / self.config["temp"], self.config["reweight"])
            loss += c_loss * (1 - self.config["em_coef"])
            record["c_loss"] = c_loss.item()

        if self.config["div_coef"] > 0: 
            d_loss = div_loss(outputs.logits, self.config["non_blank"]) 
            loss += d_loss * self.config["div_coef"]
            record["d_loss"] = d_loss.item()
        
        record["total_loss"] = loss.item()

        if self.config["l2_coef"] > 0: 
            l2_loss = self.l2_loss()
            loss += l2_loss * self.config["l2_coef"]
            record["l2_loss"] = l2_loss.item()

        self.model.zero_grad()
        loss.backward()
        # print(e_loss.item(), c_loss.item(), l2_loss.item())
        # print(predicted_ids)
        self.optimizer.step()
        if self.scheduler is not None: 
            self.scheduler.step()
        self.model.zero_grad()
    
    def suta_adapt_loss_only(self, wavs, record={}):
        """
        suta_adapt without gradient control so that we can use gradient accumulation
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        # print(type(inputs))  # inputs belongs to a custom dict class defined in transformers, not tensor
        inputs = inputs.to(device=self.model.device)
        outputs = self.model(**inputs)
        predicted_ids = torch.argmax(outputs.logits, dim=-1)

        loss = 0
        if self.config["em_coef"] > 0:
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
            x = softmax_entropy(outputs.logits / self.config["temp"])
            if self.config["non_blank"]:
                x = x[non_blank]
            if len(x) > 0:
                e_loss = x.mean(0).mean()
            else:
                e_loss = torch.tensor(0, device=self.model.device)
                record["collapse"] = True
            loss += e_loss * self.config["em_coef"]
            record["e_loss"] = e_loss.item()
        
        if 1 - self.config["em_coef"] > 0: 
            c_loss = mcc_loss(outputs.logits / self.config["temp"], self.config["reweight"])
            loss += c_loss * (1 - self.config["em_coef"])
            record["c_loss"] = c_loss.item()

        if self.config["div_coef"] > 0: 
            d_loss = div_loss(outputs.logits, self.config["non_blank"]) 
            loss += d_loss * self.config["div_coef"]
            record["d_loss"] = d_loss.item()
        
        record["total_loss"] = loss.item()

        if self.config["l2_coef"] > 0: 
            l2_loss = self.l2_loss()
            loss += l2_loss * self.config["l2_coef"]
            record["l2_loss"] = l2_loss.item()

        return loss

    def suta_adapt_auto(self, wavs, batch_size=-1, record={}) -> None:
        """ suta_adapt auto split to smaller batch """
        self.adapt_count += 1
        if batch_size == -1:
            batch_size == len(wavs)
        self.model.zero_grad()
        denom_scale = len(wavs) // batch_size
        assert denom_scale > 0
        for wavs in batchify(wavs, batch_size=batch_size):
            loss = self.suta_adapt_loss_only(wavs, record=record)
            self.adapt_count -= 1  # avoid repeat count
            loss = loss / denom_scale
            loss.backward()
    
        self.optimizer.step()
        self.model.zero_grad()

    def ctc_adapt(self, wavs, texts, record={}):
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        labels = self._text_to_model_input(texts)
        if labels.shape[1] == 0:  # empty string exception, e.g. PL collapse
            labels = torch.zeros((len(labels), 1), device=labels.device)
            record["collapse"] = True
        inputs["labels"] = labels

        outputs = self.model(**inputs)
        loss = outputs.loss
        record["ctc_loss"] = loss.item()
        record["total_loss"] = loss.item()

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None: 
            self.scheduler.step()
        self.model.zero_grad()

    def ctc_adapt_loss_only(self, wavs, texts, record={}):
        """
        ctc_adapt without gradient control so that we can use gradient accumulation
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        labels = self._text_to_model_input(texts)
        if labels.shape[1] == 0:  # empty string exception, e.g. PL collapse
            labels = torch.zeros((len(labels), 1), device=labels.device)
            record["collapse"] = True
        inputs["labels"] = labels

        outputs = self.model(**inputs)
        loss = outputs.loss
        record["ctc_loss"] = loss.item()
        record["total_loss"] = loss.item()

        return loss

    def ctc_adapt_auto(self, wavs, texts, batch_size=-1, record={}) -> None:
        """ ctc_adapt auto split to smaller batch """
        self.adapt_count += 1
        if batch_size == -1:
            batch_size == len(wavs)
        self.model.zero_grad()
        denom_scale = len(wavs) // batch_size
        assert denom_scale > 0
        for wavs, texts in zip(batchify(wavs, batch_size=batch_size), batchify(texts, batch_size=batch_size)):
            loss = self.ctc_adapt_loss_only(wavs, texts, record=record)
            self.adapt_count -= 1  # avoid repeat count
            loss = loss / denom_scale
            loss.backward()
    
        self.optimizer.step()
        self.model.zero_grad()

    # inference
    @torch.no_grad()
    def inference(self, wavs, return_logits=False):
        inputs = self._wav_to_model_input(wavs)
        outputs = self.model(**inputs).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.raw_processor_no_lm.batch_decode(predicted_ids)
        
        if return_logits:
            logits = outputs.detach().cpu().numpy()
            return list(transcription), logits
        else:
            return list(transcription)
    
    @torch.no_grad()
    def beam_inference(self, wavs, n_best=1, text_only=True):
        """ Note that the underlying model should support beam search! """
        inputs = self._wav_to_model_input(wavs)
        logits = self.model(**inputs).logits
        # CAUTION:
        # See https://www.youtube.com/watch?v=mp7fHMTnK9A for definition of alpha and beta, and note that the defualt 
        # value of beta is not 0, which includes word length penalty and therefore not pure LM score
        if len(logits) == 1:  # no batch
            res = self.processor.decode(logits[0].cpu().numpy(), n_best=n_best, alpha=0.5, beta=0.0)
        else:
            res = self.processor.batch_decode(logits.cpu().numpy(), n_best=n_best, alpha=0.5, beta=0.0)
        if not text_only:
            return res
        transcription = res.text
        
        return list(transcription)
    
    # indicators
    @torch.no_grad()
    def calc_probability(self, wavs) -> float:
        assert len(wavs) == 1
        inputs = self._wav_to_model_input(wavs)  # inputs belongs to a custom dict class defined in transformers, not tensor
        inputs = inputs.to(device=self.model.device)
        outputs = self.model(**inputs).logits
        probability = outputs.log_softmax(dim=-1).sum().item()  # all experiments do not use length normalization to maintain simplicity

        return probability

    def calc_lm_score(self, text, normalized=False) -> float:
        self.processor.decoder.reset_params(  # CAUTION: need to reset to correct parameters or else will mismatch beam search scores!
            alpha=1.0, beta=0.0, unk_score_offset=None, lm_score_boundary=None
        )
        lm = self.processor.decoder._language_model
        raw_lm_score = 0.0
        start_state = lm.get_start_state()
        words = text.split(" ")
        n_word = len(words)
        for idx in range(n_word):
            score, start_state = lm.score(start_state, words[idx], is_last_word=False)
            # print("Word: ", words[idx], score)
            raw_lm_score += score
        
        # finalize
        score, _ = lm.score(start_state, "", is_last_word=True)
        raw_lm_score += score
        if normalized:
            raw_lm_score = raw_lm_score / (n_word + 1)
        return raw_lm_score
    
    @torch.no_grad()
    def calc_suta_loss(self, wavs):
        record = {}
        self.suta_adapt_loss_only(wavs, record=record)
        self.adapt_count -= 1
        return record

    @torch.no_grad()
    def calc_ctc_loss(self, wavs, texts):
        record = {}
        self.ctc_adapt_loss_only(wavs, texts, record=record)
        self.adapt_count -= 1
        return record

    def snapshot(self, key: str):
        """Copy the model and optimizer states for resetting after adaptation."""
        # print(f"Store state. (key: {key})")
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        if self.scheduler is not None:
            scheduler_state = deepcopy(self.scheduler.state_dict())
        else:
            scheduler_state = None
        self.history[key] = (model_state, optimizer_state, scheduler_state)
    
    def load_snapshot(self, key: str) -> None:
        """Restore the model and optimizer states from copies."""
        # print(f"Reset. (key: {key})")
        model_state, optimizer_state, scheduler_state = self.history[key]
        model_state = deepcopy(model_state)
        self.model.load_state_dict(model_state, strict=True)
        
        if optimizer_state is not None:
            # optimizer_state = self.history["init"][1]
            optimizer_state = deepcopy(optimizer_state)
            self.optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            scheduler_state = deepcopy(scheduler_state)
            self.scheduler.load_state_dict(scheduler_state)

    def delete_snapshot(self, key: str) -> None:
        """Delete specific history."""
        self.history.pop(key)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
    
    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        trainable = []
        if self.config["bias_only"]:
            trainable = ['bias']
        else: 
            trainable = ['weight', 'bias']

        if self.config.get("bitfit", False):
            for np, p in self.model.named_parameters():
                if str(np).split('.')[1] == 'encoder' and "bias" in np:
                    p.requires_grad = True
                    params.append(p)
                    names.append(np)
        
        for nm, m in self.model.named_modules():
            # print(nm)
            if self.config["train_LN"]: 
                if isinstance(m, nn.LayerNorm):
                    for np, p in m.named_parameters():
                        if np in trainable:
                            if not p.requires_grad:
                                p.requires_grad = True
                                params.append(p)
                                names.append(f"{nm}.{np}")
            if self.config["train_feature"]:
                if len(str(nm).split('.')) > 1:
                    if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                        for np, p in m.named_parameters():
                            p.requires_grad = True
                            params.append(p)
                            names.append(f"{nm}.{np}")
                            
            if self.config["train_all"]: 
                for np, p in m.named_parameters():
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")

        return params, names


def setup_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    opt = getattr(torch.optim, opt_name)
    print(f'[INFO]    optimizer: {opt}')
    print(f'[INFO]    scheduler: {scheduler}')
    if opt_name == 'Adam':       
        optimizer = opt(params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)
    
    if scheduler is not None: 
        return optimizer, eval(scheduler)(optimizer, step_size=step_size, gamma=gamma)
    else: 
        return optimizer, None
