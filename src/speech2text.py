from typing import Tuple
from os import path as osp
import functools
import json
import argparse
import torch
from torch import Tensor
import ffmpeg
import numpy as np
import torchaudio
import whisper
from whisper.tokenizer import get_tokenizer
from speechbrain.pretrained import EncoderClassifier
from speechbrain.processing.speech_augmentation import Resample

INVALID_TEXTS = [
    "субтитр",
    "корректор",
    "продолжение следует",

]


class AudioNormalizer:
    """Normalizes audio into a standard format

    Arguments
    ---------
    sample_rate : int
        The sampling rate to which the incoming signals should be converted.
    mix : {"avg-to-mono", "keep"}
        "avg-to-mono" - add all channels together and normalize by number of
        channels. This also removes the channel dimension, resulting in [time]
        format tensor.
        "keep" - don't normalize channel information

    Example
    -------
    >>> import torchaudio
    >>> example_file = 'tests/samples/multi-mic/speech_-0.82918_0.55279_-0.082918.flac'
    >>> signal, sr = torchaudio.load(example_file, channels_first = False)
    >>> normalizer = AudioNormalizer(sample_rate=8000)
    >>> normalized = normalizer(signal, sr)
    >>> signal.shape
    torch.Size([160000, 4])
    >>> normalized.shape
    torch.Size([80000])

    NOTE
    ----
    This will also upsample audio. However, upsampling cannot produce meaningful
    information in the bandwidth which it adds. Generally models will not work
    well for upsampled data if they have not specifically been trained to do so.
    """

    def __init__(self, sample_rate=16000, mix="avg-to-mono"):
        self.sample_rate = sample_rate
        if mix not in ["avg-to-mono", "keep"]:
            raise ValueError(f"Unexpected mixing configuration {mix}")
        self.mix = mix
        self._cached_resample = functools.lru_cache(maxsize=12)(Resample)

    def __call__(self, audio, sample_rate):
        """Perform normalization

        Arguments
        ---------
        audio : tensor
            The input waveform torch tensor. Assuming [time, channels],
            or [time].
        """
        resampler = self._cached_resample(sample_rate, self.sample_rate)
        resampled = resampler(audio.unsqueeze(0)).squeeze(0)
        return self._mix(resampled)

    def _mix(self, audio):
        """Handle channel mixing"""
        flat_input = audio.dim() == 1
        if self.mix == "avg-to-mono":
            if flat_input:
                return audio
            return torch.mean(audio, 1)
        if self.mix == "keep":
            return audio


class LangRecognizer:
    def __init__(self, model_path: str, langs:dict):
        """
        model_path (str): /path/to/voxlingua/
        langs (dict): {"code": "language"}
        """
        super().__init__()
        self.voxlingua107 = EncoderClassifier.from_hparams(source=model_path,
                                                           savedir=model_path)
        self.audio_normalizer = AudioNormalizer()
        self.langs = langs

    def preprocess(self, audio_path: str) -> Tensor:
        if not osp.exists(audio_path):
            raise FileNotFoundError(audio_path)
        signal, sr = torchaudio.load(str(audio_path), channels_first=False)
        return self.audio_normalizer(signal, sr)
       
    def inference(self, audio_path: str) -> Tuple[str, str]:
        waveform = self.preprocess(audio_path)
        prediction = self.voxlingua107.classify_batch(waveform)
        language, lang_code = self.postprocess(prediction)
        return language, lang_code

    def postprocess(self, prediction) -> Tuple[str, str]:
        lang_predict_best = prediction[3][0]  # lang code
        language = self.langs[lang_predict_best].lower()
        return language, lang_predict_best



class MultilangSTT:
    def __init__(self, model_path:str, device="cpu"):
        super().__init__()
        self.model = whisper.load_model(model_path,
                                        device, 
                                        download_root="./models/")

    def _suppress_tokens(self) -> list:
        """Get tokens indices of integers from tokenizer

        Returns:
            list: tokens indices
        """
        tokenizer = get_tokenizer(multilingual=True)
        number_tokens = [
            i
            for i in range(tokenizer.eot)
            if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
        ]
        return number_tokens

    def preprocess(self, audio:Tensor, 
                   original_sr:int, target_sr:int=16000) -> np.array:
        try:
            audio_bytes = audio.numpy().tobytes()
            p = (
                ffmpeg.input('pipe:', format='f32le', ac=1, ar=original_sr)
                .output('pipe:', format='s16le', acodec='pcm_s16le', 
                        ac=1, ar=target_sr)
                # Add these arguments to disable verbose output
                .global_args('-hide_banner', '-loglevel', 'error')
                .run_async(pipe_stdin=True, pipe_stdout=True)
            )
            out, err = p.communicate(input=audio_bytes)
            if p.returncode != 0:
                msg = f"Failed to resample audio: {err.decode()}"
                raise RuntimeError(msg)
        except Exception as error:
            msg = f"Failed to resample audio: {str(error)}"
            raise RuntimeError(msg) from error

        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

    def transcribe(self, audio_path: str, lang: str) -> dict:
        if not osp.exists(audio_path):
            raise FileNotFoundError(audio_path)
        sup_tokens = self._suppress_tokens()
        waveform, sr = torchaudio.load(audio_path)
        replicas = []
        for channel, wav in enumerate(waveform):
            audio = self.preprocess(wav, sr)
            stt_result = self.model.transcribe(
                            audio, language=lang, 
                            suppress_tokens=[-1] + sup_tokens
                        )
            replicas += self.postprocess(stt_result, channel)
        return replicas

    @staticmethod
    def _validate_text(text:str) -> str:
        """ There may be some fake invalid texts """
        lower_text = text.lower()
        for invalid_text in INVALID_TEXTS:
            if invalid_text in lower_text:
                return
        return text

    def postprocess(self, stt_result: dict, channel: int) -> list:
        replicas = []
        for segment in stt_result['segments']:
            text = self._validate_text(segment['text'])
            if not text:
                continue
            replicas.append({'ch': channel,
                          'start': segment['start'],  # sec
                          'end': segment['end'],      # sec
                          'text': text.strip()
                          })
        return replicas


def main(args):
    models_dir = args.models_dir

    # load langs JSON
    langs_json = osp.join(models_dir, "voxlingua/langs.json")
    with open(langs_json, 'rb') as f:
        langs = json.load(f)  # dict

    # load language recognition model
    lang_code = args.lang
    if not lang_code:
        model_dir = osp.join(models_dir, "voxlingua")
        lang_rec = LangRecognizer(model_dir, langs=langs)
        _, lang_code = lang_rec.inference(args.input)

    # Multilang Speech to Text
    model_dir = osp.join(models_dir, "multilang_stt.pt")
    stt_model = MultilangSTT(model_dir, device=args.device)
    transcripts = stt_model.transcribe(args.input, lang_code)

    print(transcripts)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(transcripts, f, ensure_ascii=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Audio edotor')
    parser.add_argument('-i','--input', type=str, 
            required=True,
            help='/path/to/input/audio.wav')
    parser.add_argument('-m','--models-dir', type=str, 
            default='./models/',
            help='/path/to/dir/multilang_stt.pt /path/to/dir/voxlingua')
    parser.add_argument('-d','--device', type=str, 
            default='cpu',
            help='cpu or cuda:{X}')
    parser.add_argument('-l', '--lang', type=str, 
            choices=["ru", "en"], default=None,
            help='predefined language')
    parser.add_argument('-o','--output', type=str, 
            help='/path/to/save/transcripts.json')
    args = parser.parse_args()

    main(args)
