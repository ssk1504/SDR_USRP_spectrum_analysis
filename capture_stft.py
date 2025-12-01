import uhd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# ==========================================
# 1. PARAMETERS
# ==========================================
CENTER_FREQ = 2432e6      # Hz (2.432 GHz)
SAMPLE_RATE = 10e6         # Hz
GAIN        = 40          # dB

DWELL_TIME  = 0.2        # Seconds

STFT_NPERSEG = 256     # Can take 1024 if the system has high RAM
STFT_OVERLAP = 128     # generally 50%
WINDOW_TYPE  = 'hann'

SAVE_STFT_FILE = "stft_output.npz"
SAVE_IQ_FILE   = "captured_iq.npy"

# ==========================================
# 2. USRP HARDWARE SETUP
# ==========================================
def setup_usrp(fc, fs, gain):
    print(f"[INFO] Initializing USRP B210...")
    usrp_dev = uhd.usrp.MultiUSRP("type=b200")

    usrp_dev.set_clock_source("internal")

    print(f"[INFO] Tuning to {fc/1e6} MHz with {fs/1e6} MHz Bandwidth...")
    usrp_dev.set_rx_rate(fs, 0)
    usrp_dev.set_rx_freq(uhd.types.TuneRequest(fc), 0)
    usrp_dev.set_rx_gain(gain, 0)
    usrp_dev.set_rx_bandwidth(fs, 0)

    time.sleep(1)
    return usrp_dev

# ==========================================
# 3. CAPTURE LOGIC
# ==========================================
def capture_samples(usrp_dev, duration, fs):
    num_samps = int(np.ceil(duration * fs))

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp_dev.get_rx_stream(st_args)

    recv_buffer = np.zeros((1, 2048), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    full_data = np.zeros(num_samps, dtype=np.complex64)

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = True
    stream_cmd.time_spec = usrp_dev.get_time_now()

    print(f"[INFO] Capturing {duration} seconds of data...")
    streamer.issue_stream_cmd(stream_cmd)

    samps_received = 0

    while samps_received < num_samps:
        samps = streamer.recv(recv_buffer, metadata)

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(f"[ERROR] {metadata.strerror()}")

        end_idx = min(samps_received + samps, num_samps)
        full_data[samps_received:end_idx] = recv_buffer[0, :end_idx-samps_received]
        samps_received += samps

    print(f"[INFO] Capture Complete. Acquired {len(full_data)} samples.")

    # Save raw IQ samples
    np.save(SAVE_IQ_FILE, full_data)
    print(f"[INFO] IQ samples saved to {SAVE_IQ_FILE}")

    return full_data

# ==========================================
# 4. PLOT TIME-DOMAIN SIGNAL
# ==========================================
def plot_time_domain(iq_data, fs):
    t = np.arange(len(iq_data)) / fs

    plt.figure(figsize=(12, 5))
    plt.plot(t, iq_data.real, label='Real')
    plt.plot(t, iq_data.imag, label='Imag')
    plt.plot(t, np.abs(iq_data), label='Magnitude', alpha=0.7)

    plt.title("Time Domain IQ Samples")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. PROCESSING — Compute STFT, Save, and Plot
# ==========================================
def plot_spectrogram(data, fs, fc, nperseg, noverlap, window):
    print("[INFO] Computing STFT...")

    f, t, Zxx = signal.stft(data, fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap,
                            return_onesided=False)

    f_shifted = np.fft.fftshift(f)
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)

    Zxx_db = 20 * np.log10(np.abs(Zxx_shifted) + 1e-12)

    # SAVE STFT DATA
    np.savez(SAVE_STFT_FILE,
             frequencies=f_shifted + fc,
             time=t,
             stft_raw=Zxx_shifted,
             stft_db=Zxx_db)
    print(f"[INFO] STFT saved to {SAVE_STFT_FILE}")

    # PLOT SPECTROGRAM
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, (f_shifted + fc)/1e6, Zxx_db,
                   shading='gouraud', cmap='inferno')

    plt.title(f"Spectrogram | Fc = {fc/1e6} MHz | BW = {fs/1e6} MHz")
    plt.ylabel("Frequency (MHz)")
    plt.xlabel("Time (s)")
    cbar = plt.colorbar()
    cbar.set_label("Power (dB)")

    plt.axhline(y=fc/1e6, color='w', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        usrp = setup_usrp(CENTER_FREQ, SAMPLE_RATE, GAIN)
        iq_data = capture_samples(usrp, DWELL_TIME, SAMPLE_RATE)

        plot_time_domain(iq_data, SAMPLE_RATE)

        plot_spectrogram(iq_data, SAMPLE_RATE, CENTER_FREQ,
                         STFT_NPERSEG, STFT_OVERLAP, WINDOW_TYPE)

    except RuntimeError as e:
        print(f"[CRITICAL ERROR] {e}")
    except KeyboardInterrupt:
        print("Interrupted by user.")

