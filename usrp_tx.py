#!/usr/bin/env python3
import uhd
import numpy as np
import time
import random
from scipy.signal import chirp, hilbert

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
# CHOICES: "WIFI", "BLUETOOTH", "CHIRP"
MODE = "CHIRP"          # set to "CHIRP" to use RF-range hopping chirp

# --- USRP parameters ---
SAMPLE_RATE = 4e6      # 4 MSPS
TX_GAIN     = 50       # dB

# Default WIFI center (when MODE == "WIFI")
CENTER_FREQ_WIFI = 2432e6  # Hz

# Bluetooth hop list (in Hz)
BT_FREQ_LIST = [2402e6, 2410e6, 2420e6, 2432e6,
                2440e6, 2450e6, 2460e6, 2480e6]

# Timing (seconds)
WIFI_INTERVAL    = 0.100   # gap between chirps in WIFI mode
CHIRP_DURATION   = 0.010   # baseband chirp duration (10 ms)
BT_HOP_DWELL     = 0.05    # dwell per Bluetooth hop (50 ms)

# Baseband chirp frequencies (Hz)
# These are baseband (relative to LO). Keep them inside ±(fs/2).
CHIRP_F0 = 0        # start at DC
CHIRP_F1 = 2e6      # sweep to 2 MHz (fits in 4 MHz sampling)

# === CHIRP (RF sweep) parameters (for MODE == "CHIRP") ===
CHIRP_RF_START = 2430e6   # RF start (Hz)
CHIRP_RF_END   = 2434e6   # RF end (Hz)
CHIRP_RF_STEP  = 1e6      # step in Hz (1 MHz steps; tune to taste)
CHIRP_RF_DWELL = 0.05     # how long to transmit at each RF hop (s)

# If you want continuous RF sweep/back-and-forth, set this True (will loop up & down)
CHIRP_RF_PENDULUM = False

# Power scaling for chirp
CHIRP_AMPLITUDE = 0.7

# ==========================================
# 2. WAVEFORM GENERATION
# ==========================================
def generate_chirp(fs, duration, f0, f1, amplitude=0.7):
    """
    Generate analytic (complex) chirp using Hilbert transform.
    Returns complex64 IQ samples.
    """
    num_samps = int(fs * duration)
    if num_samps <= 0:
        raise ValueError("Duration too small or sample rate too small -> zero samples.")
    t = np.arange(num_samps) / fs

    # Real-valued linear chirp
    x = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')

    # Convert to complex analytic signal (so TX sees IQ)
    x_complex = hilbert(x) * amplitude
    return x_complex.astype(np.complex64)

def generate_bt_tone(fs, duration):
    num_samps = int(fs * duration)
    t = np.arange(num_samps)
    freq_offset = 500e3  # 500 kHz offset to avoid DC
    tone = 0.7 * np.exp(1j * 2 * np.pi * freq_offset * t / fs)
    return tone.astype(np.complex64)

# ==========================================
# 3. USRP SETUP
# ==========================================
def setup_usrp_tx():
    print("[INFO] Initializing USRP for Transmission...")
    usrp = uhd.usrp.MultiUSRP("type=b200")
    
    # Configure Tx chain
    usrp.set_tx_rate(SAMPLE_RATE, 0)
    usrp.set_tx_gain(TX_GAIN, 0)
    usrp.set_tx_bandwidth(SAMPLE_RATE, 0)

    # === ADDED: Explicitly set Antenna ===
    usrp.set_tx_antenna("TX/RX", 0)
    print(f"[INFO] Antenna set to: {usrp.get_tx_antenna(0)}")
    
    # small settle
    time.sleep(1)
    return usrp

def transmit_burst(streamer, samples, metadata):
    """
    send samples as a burst. 
    Reshapes samples to (1, N) to satisfy UHD requirement.
    """
    # === FIX: Ensure array is 2D (1, num_samples) ===
    if len(samples.shape) == 1:
        samples = samples.reshape(1, -1)
    
    metadata.start_of_burst = True
    metadata.end_of_burst = False
    metadata.has_time_spec = False
    
    streamer.send(samples, metadata)
    
    metadata.start_of_burst = False
    metadata.end_of_burst = True
    
    # Send EOB (End of Burst). 
    # Must also be 2D, even for zero length.
    streamer.send(np.zeros((1, 0), dtype=np.complex64), metadata)

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    try:
        # Setup USRP
        usrp_dev = setup_usrp_tx()
        
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")  # float32 complex in host, 16-bit on wire
        streamer = usrp_dev.get_tx_stream(st_args)
        metadata = uhd.types.TXMetadata()
        
        # Generate baseband waveforms
        chirp_samples = generate_chirp(SAMPLE_RATE, CHIRP_DURATION, CHIRP_F0, CHIRP_F1, CHIRP_AMPLITUDE)
        bt_samples    = generate_bt_tone(SAMPLE_RATE, BT_HOP_DWELL)
        
        print(f"[TX] Mode: {MODE}")
        print(f"[TX] Sample rate: {SAMPLE_RATE/1e6} MSPS, Gain: {TX_GAIN} dB")
        print("[INFO] Press Ctrl+C to stop.\n")
        
        # -----------------------------
        # WIFI MODE (baseband chirp bursts)
        # -----------------------------
        if MODE == "WIFI":
            usrp_dev.set_tx_freq(uhd.types.TuneRequest(CENTER_FREQ_WIFI), 0)
            time.sleep(0.5)
            print(f"-> WIFI TX LO set to {CENTER_FREQ_WIFI/1e6} MHz")
            
            while True:
                transmit_burst(streamer, chirp_samples, metadata)
                time.sleep(WIFI_INTERVAL)

        # -----------------------------
        # BLUETOOTH MODE (FHSS tone)
        # -----------------------------
        elif MODE == "BLUETOOTH":
            print("-> Bluetooth FHSS Mode")
            while True:
                freq = random.choice(BT_FREQ_LIST)
                usrp_dev.set_tx_freq(uhd.types.TuneRequest(freq), 0)
                
                transmit_burst(streamer, bt_samples, metadata)
                print(f" -> Hopped to {freq/1e6} MHz")
                time.sleep(0.01)

        # -----------------------------
        # CHIRP MODE: hop LO from CHIRP_RF_START -> CHIRP_RF_END
        # and transmit baseband chirp at each LO.
        # -----------------------------
        elif MODE == "CHIRP":
            # build list of RF center freqs
            if CHIRP_RF_STEP <= 0:
                raise ValueError("CHIRP_RF_STEP must be > 0")
            freqs = np.arange(CHIRP_RF_START, CHIRP_RF_END + CHIRP_RF_STEP/2, CHIRP_RF_STEP)

            if len(freqs) == 0:
                raise ValueError("No RF frequencies generated. Check start/end/step.")

            print(f"-> CHIRP mode RF sweep from {CHIRP_RF_START/1e6:.3f} MHz to {CHIRP_RF_END/1e6:.3f} MHz")
            print(f"   step = {CHIRP_RF_STEP/1e6:.3f} MHz, dwell = {CHIRP_RF_DWELL}s, points = {len(freqs)}")

            # Optionally pendulum (go up and down)
            up = True
            idx = 0
            # Loop forever, hopping LO and transmitting the chirp at each hop
            while True:
                # determine RF freq index
                if CHIRP_RF_PENDULUM:
                    # pendulum pattern: 0..N-1, N-2..1, repeat
                    sequence = list(freqs) + list(freqs[-2:0:-1]) if len(freqs) > 1 else list(freqs)
                    for rf in sequence:
                        usrp_dev.set_tx_freq(uhd.types.TuneRequest(float(rf)), 0)
                        time.sleep(0.01)  # small settle between retunes
                        transmit_burst(streamer, chirp_samples, metadata)
                        print(f" -> Transmitted chirp at LO {rf/1e6:.3f} MHz")
                        time.sleep(CHIRP_RF_DWELL)
                else:
                    # simple wrap-around sweep
                    for rf in freqs:
                        usrp_dev.set_tx_freq(uhd.types.TuneRequest(float(rf)), 0)
                        time.sleep(0.01)  # allow LO settle (increase if needed)
                        transmit_burst(streamer, chirp_samples, metadata)
                        print(f" -> Transmitted chirp at LO {rf/1e6:.3f} MHz")
                        time.sleep(CHIRP_RF_DWELL)

        else:
            print(f"[ERROR] Unknown MODE '{MODE}'. Choose WIFI, BLUETOOTH or CHIRP.")

    except RuntimeError as e:
        print(f"[ERROR] RuntimeError: {e}")
    except KeyboardInterrupt:
        print("\n[STOP] Transmission stopped by user.")
    except Exception as ex:
        print(f"[ERROR] Unexpected exception: {ex}")
