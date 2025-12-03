import numpy as np
import adi
import time
import random

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
SDR_IP = "ip:192.168.2.1" # Address of the Pluto

# CHOOSE YOUR MODE HERE:
# "WIFI"      -> Simulates periodic Beacon Frames (Broadband bursts)
# "BLUETOOTH" -> Simulates Frequency Hopping (Narrowband hops)
MODE = "WIFI" 

# Common Parameters
SAMPLE_RATE = 4e6         # 4 MSPS (Safe for USB 2.0 limit)
TX_GAIN     = -20         # dB (Range: 0 to -89). -20 is moderate power. 
                          # WARNING: Do not set to 0 if antennas are close!

# ------------------------------------------
# Mode A: Wi-Fi Beacon Configuration
# ------------------------------------------
WIFI_CENTER_FREQ = 2432e6 # Hz (Matches your USRP Rx)
WIFI_INTERVAL    = 0.1024 # Seconds (Standard 102.4 ms beacon interval)
WIFI_BURST_LEN   = 0.002  # Seconds (2ms burst duration)

# ------------------------------------------
# Mode B: Bluetooth Hopper Configuration
# ------------------------------------------
# Bluetooth range is 2402-2480 MHz.
# We will hop randomly within this list.
BT_FREQ_LIST = [2402e6, 2410e6, 2420e6, 2432e6, 2440e6, 2450e6, 2460e6, 2480e6]
HOP_DWELL_TIME = 0.05     # Seconds per hop (Fast enough to look cool, slow enough for Python)

# ==========================================
# 2. SIGNAL GENERATION FUNCTIONS
# ==========================================
def generate_wifi_waveform(fs, total_time, burst_time):
    """
    Creates a frame that is mostly empty (0) with a short burst of noise.
    When looped, this looks exactly like a periodic beacon.
    """
    total_samples = int(fs * total_time)
    burst_samples = int(fs * burst_time)
    
    # 1. Create the empty container
    buffer = np.zeros(total_samples, dtype=np.complex64)
    
    # 2. Generate random QPSK-like noise for the burst
    # (Real Wi-Fi is OFDM, but wideband noise looks similar on a spectrogram)
    noise = (np.random.randn(burst_samples) + 1j * np.random.randn(burst_samples)) / np.sqrt(2)
    
    # 3. Insert the burst at the beginning
    buffer[:burst_samples] = noise
    
    # Scale to max amplitude (0.5 to prevent clipping in DAC)
    buffer = buffer * 0.5
    return buffer

def generate_bt_waveform(fs):
    """
    Creates a continuous narrowband signal.
    We will shift the Center Freq (LO) physically to make it 'hop'.
    """
    # A small buffer of continuous signal (Gaussian Frequency Shift Keying sim)
    num_samples = 10000 
    
    # Narrowband signal (concentrated energy)
    t = np.arange(num_samples)
    freq_offset = 500e3 # 500 kHz offset from center
    
    # Generate a complex sine wave (tone)
    signal = 0.5 * np.exp(1j * 2 * np.pi * freq_offset * t / fs)
    return signal.astype(np.complex64)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def run_transmitter():
    print(f"[INFO] Connecting to Pluto at {SDR_IP}...")
    sdr = adi.Pluto(SDR_IP)
    
    # Configure General Settings
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.tx_rf_bandwidth = int(SAMPLE_RATE) # Match sample rate
    sdr.tx_hardwaregain_chan0 = int(TX_GAIN)
    
    print(f"[INFO] Mode Selected: {MODE}")

    # --------------------------------------
    # EXECUTE WI-FI SIMULATION
    # --------------------------------------
    if MODE == "WIFI":
        print(f"[SETUP] Configuring for {WIFI_CENTER_FREQ/1e6} MHz...")
        sdr.tx_lo = int(WIFI_CENTER_FREQ)
        
        # Generate the bursty frame
        samples = generate_wifi_waveform(SAMPLE_RATE, WIFI_INTERVAL, WIFI_BURST_LEN)
        
        print(f"[TX] Transmitting 'Beacon' every {WIFI_INTERVAL*1000} ms...")
        print("[INFO] Press Ctrl+C to stop.")
        
        # Enable Cyclic Buffer (Hardware repeats the waveform automatically)
        sdr.tx_cyclic_buffer = True 
        sdr.tx(samples) 
        
        # Keep script alive while hardware does the work
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[STOP] Stopping Transmitter.")
            sdr.tx_destroy_buffer()

    # --------------------------------------
    # EXECUTE BLUETOOTH SIMULATION
    # --------------------------------------
    elif MODE == "BLUETOOTH":
        print(f"[SETUP] Configuring Hopping Sequence...")
        
        # Create a continuous tone
        samples = generate_bt_waveform(SAMPLE_RATE)
        
        # Enable Cyclic Buffer for the BASEBAND signal
        # We will change the CARRIER (LO) in the loop
        sdr.tx_cyclic_buffer = True
        sdr.tx(samples)
        
        print(f"[TX] Hopping through frequencies: {[f/1e6 for f in BT_FREQ_LIST]}")
        print("[INFO] Press Ctrl+C to stop.")
        
        try:
            while True:
                # Pick a random frequency
                next_freq = random.choice(BT_FREQ_LIST)
                
                # Retune the radio
                sdr.tx_lo = int(next_freq)
                print(f" -> Hopped to {next_freq/1e6} MHz")
                
                # Wait before next hop
                time.sleep(HOP_DWELL_TIME)
                
        except KeyboardInterrupt:
            print("\n[STOP] Stopping Transmitter.")
            sdr.tx_destroy_buffer()

if __name__ == "__main__":
    run_transmitter()