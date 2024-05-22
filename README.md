WAV Processing
==================
### Audio effects
```bash
python3 src/audio_effects.py -i audios/eng.wav -s 0.3 -v 2.5 -o audios/out.wav
```
* `-i/--input` - path to input audio.wav
* `-s/--speed` - speed factor > 0.0. Factor by which to adjust speed of input. Values greater than 1.0 compress waveform in time, whereas values less than 1.0 stretch waveform in time
* `-v/--volume` - volume factor > 0.0. Factor by which to adjust volume of input. Values greater than 1.0 make audio louder, whereas values less than 1.0 make audio quieter
* `-o/--output` - path where to save output audio.wav

### Speech to text
```bash
python3 src/audio_effects.py -i audios/eng.wav -m ./models/ -o transcipts.json -l ru
```
* `-i/--input` - path to input audio.wav. **Only rus or eng**
* `-m/--models-dir` - path to models dir. See descriptions below
* `-d/--device` - where to do computation. `cpu` or `cuda:{x}`
* `-l/--lang` - predefined language `ru` or `en`. If not stated, language recognition will be performed
* `-o/--output` - path where to save JSON with transcipts. Optionally

### Models dir
Models dir must have following structure
```yaml
- multilang_stt.pt
- voxligua:
    - classigier.ckpt
    - config.json
    - embedding_model.ckpt
    - hyperparams.yaml
    - label_encoder.ckpt
    - label_encoder.txt
    - langs.json
    - normalizer.ckpt
```

models can be downloaded from [here](https://drive.google.com/file/d/1c4wxKIlcWR4N3E9PU4LywDQLTooUj2KR/view?usp=sharing)