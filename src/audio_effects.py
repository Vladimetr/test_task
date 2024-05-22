from torch import Tensor
import torchaudio
import argparse


def apply_effects(sample:Tensor, sr:int,
                  speed_factor:float=1.0,
                  volume_factor:float=1.0) -> Tensor:
    """
    Apply audio effects
    Args:
        sample (CH, S): input sample (waveform)
        speed_factor (float): values greater than 1.0 
                compress waveform in time, whereas values 
                less than 1.0 stretch waveform in time.
                NOTE: value must be positive.
        volume_factor (float): values greater than 1.0 
                make audio louder, whereas values 
                less than 1.0 make audio quieter
                NOTE: value must be positive.
	Returns:
		(CH, S'): output sample (waveform).
        NOTE: S' > S if speed_factor < 1.0
			  S' < S if speed_factor > 1.0
    """
    sox_effects = [
        ["speed", str(speed_factor)],
        ["vol", str(volume_factor)],
        ["rate", str(sr)],

    ]
    sample_out, _ = torchaudio.sox_effects.apply_effects_tensor(
        sample, sr, sox_effects)
    
    return sample_out


def main(args):
    # load audio
    sample, sr = torchaudio.load(args.input)

    sample = apply_effects(sample, sr, 
                           speed_factor=args.speed, 
                           volume_factor=args.volume)

    # save audio
    torchaudio.save(args.output, sample, sr)
    print(f"Resulted audio saved: '{args.output}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Audio edotor')
    parser.add_argument('-i','--input', type=str, 
            required=True,
            help='/path/to/input/audio.wav')
    parser.add_argument('-s', '--speed', type=float,
            default=1.0,
            help="Factor by which to adjust speed of input. "
            "Values greater than 1.0 compress waveform in time, "
            "whereas values less than 1.0 stretch waveform in time")
    parser.add_argument('-v','--volume', type=float, 
            default=1.0,
            help="Factor by which to adjust volume of input. "
            "Values greater than 1.0 make audio louder, "
            "whereas values less than 1.0 make audio quieter")
    parser.add_argument('-o','--output', type=str, 
            required=True,
            help='/path/to/output/audio.wav')
    args = parser.parse_args()

    main(args)

