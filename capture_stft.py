import uhd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# ==========================================
# 1. PARAMETERS
# ==========================================
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
    
    # Wait for LO to lock
    time.sleep(1)
    return usrp_dev

# ==========================================
# 3. CAPTURE LOGIC
# ==========================================
def capture_samples(usrp_dev, duration, fs):
    num_samps = int(np.ceil(duration * fs))
    
    # Create the streamer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp_dev.get_rx_stream(st_args)
    
    # Pre-allocate the full buffer (Complex Float32)
    # 0.5s @ 20MHz = 10M samples = ~80MB RAM (Safe)
    full_data = np.zeros(num_samps, dtype=np.complex64)
    
    # Buffer for individual packet reads
    recv_buffer = np.zeros((1, 4096), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = True
    stream_cmd.time_spec = usrp_dev.get_time_now()

    print(f"[INFO] Capturing {duration} seconds of data...")
    streamer.issue_stream_cmd(stream_cmd)

    samps_received = 0
    timeout_counter = 0

    while samps_received < num_samps:
        samps = streamer.recv(recv_buffer, metadata)

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(f"[ERROR] {metadata.strerror()}")
            # Break if serious error to prevent infinite loop
            if metadata.error_code != uhd.types.RXMetadataErrorCode.overflow:
                break

        if samps > 0:
            end_idx = min(samps_received + samps, num_samps)
            full_data[samps_received:end_idx] = recv_buffer[0, :end_idx-samps_received]
            samps_received += samps
            timeout_counter = 0
        else:
            timeout_counter += 1
            if timeout_counter > 1000:
                print("[WARN] Timeout waiting for samples.")
                break

    print(f"[INFO] Capture Complete. Acquired {len(full_data)} samples.")
    return full_data

# ==========================================
# 4. PLOT TIME-DOMAIN (Optimized)
# ==========================================
def plot_time_domain(iq_data, fs, label):
    # Downsample for plotting to save memory (plot 1 out of every 100 samples)
    decimation_factor = 100 
    subset = iq_data[::decimation_factor]
    t = np.arange(len(subset)) * decimation_factor / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, np.abs(subset), label='Magnitude', color='darkblue', alpha=0.8)
    plt.title(f"Time Domain (Decimated x{decimation_factor}) - {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    print("[DISPLAY] Plotting Time Domain...")
    plt.show()

# ==========================================
# 5. EFFICIENT BLOCK PROCESSING STFT
# ==========================================
def compute_and_plot_block_stft(data, fs, fc, n_fft, label):
    print("[INFO] Starting Block-Based STFT Computation...")
    
    # 50% Overlap
    hop_length = n_fft // 2
    
    # Calculate dimensions
    n_samples = len(data)
    n_frames = (n_samples - n_fft) // hop_length + 1
    
    if n_frames <= 0:
        print("[ERROR] Data too short for this STFT size.")
        return

    # Window Function (Hanning)
    window = np.hanning(n_fft)
    
    # PRE-ALLOCATE OUTPUT MATRIX (Float32 to save RAM)
    # Rows = Frequency Bins (n_fft), Cols = Time Steps (n_frames)
    # We use fftshift structure, so rows = n_fft
    spectrogram_data = np.zeros((n_fft, n_frames), dtype=np.float32)
    
    print(f"[INFO] Matrix Size: {n_fft} x {n_frames} (approx {n_fft * n_frames * 4 / 1024**2:.2f} MB)")

    # --- BLOCK PROCESSING LOOP ---
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        
        # 1. Slice
        chunk = data[start:end]
        
        # 2. Window
        windowed = chunk * window
        
        # 3. FFT
        fft_res = np.fft.fft(windowed)
        
        # 4. Shift (so 0Hz is in center)
        fft_shifted = np.fft.fftshift(fft_res)
        
        # 5. Magnitude & Log (dB) - Discard Phase IMMEDIATELY
        # Adding 1e-12 to avoid log(0)
        mag_db = 20 * np.log10(np.abs(fft_shifted) + 1e-12)
        
        # 6. Store
        spectrogram_data[:, i] = mag_db

    print("[INFO] Computation done. Saving and Plotting...")

    # OPTIONAL: Save compressed
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"stft_data_{label}_{timestamp}.npz"
    np.savez_compressed(filename, stft_db=spectrogram_data, fs=fs, fc=fc)
    print(f"[INFO] Saved to {filename}")

    # PLOT USING IMSHOW (Efficient Bitmap)
    # Extent defines the axes: [time_start, time_end, freq_start, freq_end]
    duration = n_samples / fs
    freq_min = (fc - fs/2) / 1e6
    freq_max = (fc + fs/2) / 1e6
    
    plt.figure(figsize=(10, 6))
    
    # 'aspect="auto"' allows the pixels to stretch to fill the window
    # 'origin="lower"' puts low frequencies at bottom (standard for spectrograms)
    plt.imshow(spectrogram_data, 
               aspect='auto', 
               origin='lower', 
               extent=[0, duration, freq_min, freq_max],
               cmap='inferno',
               interpolation='nearest') # 'nearest' is fastest

    plt.colorbar(label='Power (dB)')
    plt.title(f"Spectrogram | Fc={fc/1e6} MHz | BW={fs/1e6} MHz")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    
    print("[DISPLAY] Plotting Spectrogram... Close window to continue.")
    plt.show()

# ==========================================
# WRAPPER & MAIN
# ==========================================
def run_experiment(usrp_dev, fc, bw, dwell, stft_n, label):
    print(f"\n[TASK] Starting Experiment: {label}")
    
    # 1. Re-configure USRP
    usrp_dev.set_rx_rate(bw, 0)
    usrp_dev.set_rx_freq(uhd.types.TuneRequest(fc), 0)
    usrp_dev.set_rx_bandwidth(bw, 0)
    time.sleep(1.0) # Let LO settle
    
    # 2. Capture
    iq_data = capture_samples(usrp_dev, dwell, bw)
    
    # 3. Plot Time Domain
    plot_time_domain(iq_data, bw, label)
    
    # 4. Process STFT (Block Method)
    compute_and_plot_block_stft(iq_data, bw, fc, stft_n, label)

if __name__ == "__main__":
    try:
        # Initial Safe Defaults
        usrp = setup_usrp(2400e6, 5e6, 40)
        
        print("\n" + "="*40)
        print("  OPTIMIZED USRP SPECTROGRAM VIEWER")
        print("="*40)
        
        while True:
            print("\n--- NEW RUN SETUP ---")
            try:
                raw_fc = float(input("Enter Center Freq in GHz (e.g. 2.432) or 0 to STOP: "))
                if raw_fc == 0: break
                raw_bw = float(input("Enter Bandwidth in MHz (e.g. 10): "))
                dwell = float(input("Enter Dwell Time in seconds (e.g. 0.5): "))
                stft_n = int(input("Enter STFT Size (e.g. 512): "))
            except ValueError:
                print("Invalid input.")
                continue
            
            fc = raw_fc * 1e9
            bw = raw_bw * 1e6
            label = f"Exp_{raw_fc}GHz"
            
            run_experiment(usrp, fc, bw, dwell, stft_n, label)
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
