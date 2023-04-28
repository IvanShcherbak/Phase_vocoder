import librosa as lb
import numpy as np
import soundfile
import argparse


def stretch(data: np.ndarray, time_stretch_ratio: float) -> np.ndarray:
    """Phase vocoder - stretch audio sequence.
    :param data: np.ndarray, audio array
    :param time_stretch_ratio: float, coefficient of stretch
    :return: np.ndarray, time stretched audio array
    """
    nfft = 2048
    hop = nfft/4
    stft = lb.core.stft(data, n_fft=nfft)
    stft_cols = stft.shape[1]
    times = np.arange(0, stft.shape[0], 1/time_stretch_ratio)
    stft_new = np.zeros((len(times), stft_cols), dtype=np.complex_)
    phase_new = (2 * np.pi * hop * np.arange(0, stft_cols))/nfft
    phase = np.angle(stft[0])
    stft = np.concatenate(stft, np.zeros((1, stft_cols)), axis=0)

    for i, time in enumerate(times):
        left_frame = int(np.floor(time))
        local_frames = stft[[left_frame, left_frame + 1], :]
        right_wt = time - np.floor(time)
        local_mag = (1 - right_wt) * np.absolute(local_frames[0, :]) + right_wt * np.absolute(local_frames[1, :])
        local_dphi = np.angle(local_frames[1, :]) - np.angle(local_frames[0, :]) - phase_new
        local_dphi = local_dphi - 2 * np.pi * np.floor(local_dphi/(2 * np.pi))
        stft_new[i, :] = local_mag * np.exp(phase*1j)
        phase += local_dphi + phase_new

    return lb.core.istft(stft_new)


def main(arguments):
    data, samplerate = lb.core.load(arguments.input_file)
    time_stretch = stretch(data, arguments.time_stretch_ratio)
    output_wav = arguments.output_file
    soundfile.write(output_wav, time_stretch, samplerate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Растягивание или сжимание аудио без изменения тона-Phase Vocoder')
    parser.add_argument('input_file',
                        type=str,
                        help='Путь для исходного wav файла')
    parser.add_argument('output_file',
                        type=str,
                        help='Путь до файла, который является результатом работы программы')
    parser.add_argument('time_stretch_ratio',
                        type=float,
                        help='Параметр r алгоритма. 0 < r < 1 - сжатие, r >= 1 - растягивание')
    arguments = parser.parse_args()
    main(arguments)