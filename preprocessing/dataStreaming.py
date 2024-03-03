import argparse
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, AggOperations, WaveletTypes, NoiseEstimationLevelTypes, \
    WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes, WindowOperations, DetrendOperations


def main():
    BoardShim.enable_dev_board_logger()

    # parser = argparse.ArgumentParser()
    # # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    # parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
    #                     default=0)
    # parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    # parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
    #                     default=0)
    # parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    # parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='/dev/tty.usbmodem11')
    # parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    # parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    # parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    # parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
    #                     required=False, default=BoardIds.GANGLION_BOARD)
    # parser.add_argument('--file', type=str, help='file', required=False, default='')
    # parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
    #                     required=False, default=BoardIds.GANGLION_BOARD)
    # args = parser.parse_args()

    # params = BrainFlowInputParams()
    # params.ip_port = args.ip_port
    # params.serial_port = args.serial_port
    # params.mac_address = args.mac_address
    # params.other_info = args.other_info
    # params.serial_number = args.serial_number
    # params.ip_address = args.ip_address
    # params.ip_protocol = args.ip_protocol
    # params.timeout = args.timeout
    # params.file = args.file
    # params.master_board = args.master_board

    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbmodem11"
    params.timeout = 15

    board_id = BoardIds.GANGLION_BOARD.value
    board_descr = BoardShim.get_board_descr(board_id)
    sampling_rate = int(board_descr['sampling_rate'])
    
    board = BoardShim(BoardIds.GANGLION_BOARD, params)
    board.prepare_session()
    board.start_stream ()

    time.sleep(2)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

    board.stop_stream()
    board.release_session()

    print(data)

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    df = pd.DataFrame(np.transpose(data))
    plt.figure()
    df[eeg_channels].plot(subplots=True)
    plt.savefig('before_processing.png')

    # demo for denoising, apply different methods to different channels for demo
    for count, channel in enumerate(eeg_channels):
        # first of all you can try simple moving median or moving average with different window size
        if count == 0:
            DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEAN.value)
        elif count == 1:
            DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEDIAN.value)
        # if methods above dont work for your signal you can try wavelet based denoising
        # feel free to try different parameters
        else:
            DataFilter.perform_wavelet_denoising(data[channel], WaveletTypes.BIOR3_9, 3,
                                                 WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
                                                 WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)

    df = pd.DataFrame(np.transpose(data))
    plt.figure()
    df[eeg_channels].plot(subplots=True)
    plt.savefig('after_processing.png')


    eeg_channels = board_descr['eeg_channels']
    eeg_channel = eeg_channels[1]
    # optional detrend
    DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
    psd = DataFilter.get_psd_welch(data[eeg_channel], nfft, nfft // 2, sampling_rate,
                                   WindowOperations.BLACKMAN_HARRIS.value)

    #delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and gamma (30–100 Hz)
    band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)
    band_power_beta = DataFilter.get_band_power(psd, 14.0, 30.0)
    band_power_gamma= DataFilter.get_band_power(psd, 30.0, 100.0)
    band_power_delta = DataFilter.get_band_power(psd, 0.5, 4.0)
    # band_powers = {band_power_alpha,band_power_beta, band_power_delta, band_power_gamma}
    # plt.figure()
    # df[band_powers].plot(subplots=True)
    # plt.savefig('bandpower.png')

    print(band_power_alpha)
    print(band_power_beta)
    print(band_power_delta)
    print(band_power_gamma)


if __name__ == "__main__":
    main()