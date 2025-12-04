import uhd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from scipy.signal import windows

# ==========================================
# 1. PARAMETERS
# ==========================================
SAVE_IQ_FILE    = "captured_iq.npy"

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
    
    # --- BUILT-IN HARDWARE CALIBRATION ---
    # Enables the B210's internal Automatic DC Offset Compensation. (Notch Filter)
    # Reference: UHD Manual on self calibration
    usrp_dev.set_rx_dc_offset(True, 0)
    
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
    
    # Pre-allocate buffer
    full_data = np.zeros(num_samps, dtype=np.complex64)
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
    decimation_factor = 100 
    subset = iq_data[::decimation_factor]
    t = np.arange(len(subset)) * decimation_factor / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, np.abs(subset), label='Magnitude', color='darkblue', alpha=0.8)
    plt.title(f"Time Domain (Decimated x{decimation_factor}) - {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. EFFICIENT BLOCK PROCESSING STFT
# ==========================================
def compute_and_plot_block_stft(data, fs, fc, label):
    print("[INFO] Starting Block-Based STFT Computation...")
    
    # --- REQUIREMENT: 25 kHz Resolution ---
    # N_fft = fs / 25000
    n_fft = int(fs / 25000)
    
    # --- REQUIREMENT: 50% Overlap ---
    # Satisfies Constant Overlap-Add (COLA) constraint
    hop_length = n_fft // 2
    
    # Sanity Check for Terminal
    actual_res = fs / n_fft
    print(f"[INFO] Dynamic FFT Size: {n_fft} bins")
    print(f"[INFO] Frequency Resolution: {actual_res/1000:.2f} kHz (Target: 25.00 kHz)")
    
    n_samples = len(data)
    n_frames = (n_samples - n_fft) // hop_length + 1
    
    if n_frames <= 0:
        print("[ERROR] Data too short for this STFT size.")
        return

    # --- REQUIREMENT: Minimize Spectral Leakage ---
    # Blackman-Harris Window (Side lobes < -92 dB)
    window = windows.kaiser(n_fft, beta=14)
    #window = windows.blackmanharris(n_fft)
    
    spectrogram_data = np.zeros((n_fft, n_frames), dtype=np.float32)
    
    print(f"[INFO] Matrix Size: {n_fft} x {n_frames}")

    # --- BLOCK PROCESSING LOOP ---
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        
        chunk = data[start:end]
        windowed = chunk * window
        fft_res = np.fft.fft(windowed)
        fft_shifted = np.fft.fftshift(fft_res)
        
        # Magnitude & Log (dB)
        mag_db = 20 * np.log10(np.abs(fft_shifted) + 1e-12)
        spectrogram_data[:, i] = mag_db

    print("[INFO] Computation done. Plotting...")

    # Plot
    duration = n_samples / fs
    freq_min = (fc - fs/2) / 1e6
    freq_max = (fc + fs/2) / 1e6
    
    plt.figure(figsize=(12, 7))
    plt.imshow(spectrogram_data, 
               aspect='auto', 
               origin='lower', 
               extent=[0, duration, freq_min, freq_max],
               cmap='inferno',
               interpolation='nearest')

    plt.colorbar(label='Power (dB)')
    plt.title(f"Spectrogram | Fc={fc/1e6} MHz | BW={fs/1e6} MHz | Res={actual_res/1000:.2f} kHz")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    plt.show()

# ==========================================
# WRAPPER & MAIN
# ==========================================
def run_experiment(usrp_dev, fc, bw, dwell, label):
    print(f"\n[TASK] Starting Experiment: {label}")
    
    # --- LO OFFSET ---
    # Keeps the physical LO 6-15 MHz away from our band of interest.
    lo_offset = 15e6
    tune_req = uhd.types.TuneRequest(fc, lo_offset)

    usrp_dev.set_rx_rate(bw, 0)
    usrp_dev.set_rx_freq(tune_req, 0)
    usrp_dev.set_rx_bandwidth(bw, 0)
    time.sleep(1.0) 
    
    # 1. Capture
    iq_data = capture_samples(usrp_dev, dwell, bw)
    
    # --- STRATEGY 3: CUSTOM SOFTWARE DSP BLOCK (COMMENTED OUT) ---
    # Calculates the mean (DC component) and subtracts it.
    # Reference: Manual Pg 3-4 (Additive Error Model)
    dc_value = np.mean(iq_data)
    iq_data = iq_data - dc_value
    print(f"[DSP] Custom Software DC Removal Applied. Estimated Bias: {dc_value:.4f}")
    
    # 2. Plot Time Domain
    plot_time_domain(iq_data, bw, label)
    
    # 3. STFT
    compute_and_plot_block_stft(iq_data, bw, fc, label)

if __name__ == "__main__":
    try:
        # Default to Bluetooth Center Channel (2440 MHz) and High BW (20 MHz)
        usrp = setup_usrp(2440e6, 20e6, 30)
        
        print("\n" + "="*40)
        print("  OPTIMIZED USRP VIEWER (Triple DC-Removal Mode)")
        print("="*40)
        
        while True:
            print("\n--- NEW RUN SETUP ---")
            try:
                raw_fc = float(input("Enter Center Freq in GHz (e.g. 2.440) or 0 to STOP: "))
                if raw_fc == 0: break
                raw_bw = float(input("Enter Bandwidth in MHz (e.g. 5): "))
                dwell = float(input("Enter Dwell Time in seconds (e.g. 0.1): "))
                
            except ValueError:
                print("Invalid input.")
                continue
            
            fc = raw_fc * 1e9
            bw = raw_bw * 1e6
            label = f"Exp_{raw_fc}GHz"
            
            run_experiment(usrp, fc, bw, dwell, label)
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
