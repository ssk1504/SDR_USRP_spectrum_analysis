# UHD Python API Installation and USRP Setup

To install the UHD Python API from source, Ubuntu **22.04** and Python **3.8 or higher** are required.  
Follow the steps below to compile UHD, install the Python bindings, and configure USB permissions for USRP B200/B210 devices.

---
Follow the commands exactly as given.
---

## 1. Install Dependencies

```bash
sudo apt update
sudo apt install -y \
    git cmake g++ python3 python3-dev python3-pip \
    libboost-all-dev libusb-1.0-0-dev libncurses5-dev \
    python3-mako libaio-dev
```

## 2. Clone the UHD Repository
```bash
cd ~
git clone https://github.com/EttusResearch/uhd.git
mv uhd uhd_repo
```

## 3. Build UHD From Source
```bash
cd ~/uhd_repo/host
mkdir build
cd build

cmake \
  -DENABLE_PYTHON_API=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DPYTHON3_SITEPKG=/usr/local/lib/python3.10/dist-packages \
  ..

make -j$(( $(nproc) / 2 ))
sudo make install
sudo ldconfig
```

## 4. Install the UHD Python Module
```bash
cd ~/uhd_repo/host/build/python
sudo python3 setup.py install
```
If there are no errors, then the UHD module installed successfully

## 5. Configure USRP USB Permissions
USB 3.0/3.1 must be used for B210/B200 devices.

### Copy the Ettus udev rule file:
```bash
sudo cp ~/uhd_repo/host/utils/uhd-usrp.rules /etc/udev/rules.d/
```

### Reload udev rules:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Add current user to required groups:
```bash
sudo usermod -aG plugdev $USER
sudo usermod -aG dialout $USER
```

### Reboot the system
```bash
sudo reboot now
```

## 6. Verify UHD Installation
After reboot:
```bash
uhd_find_devices
```

Expected output:
```bash
-- UHD Device 0 --
    serial: 32A0FD4
    product: B200
    type: b200

```

## 7. Verify Python UHD Bindings
```bash
python3 - << 'EOF'
import uhd
print("UHD Python API is working")
print("UHD Version:", uhd.get_version_string())
EOF
```

Expected output:
```bash
UHD Python API is working
UHD Version: 4.x.x
```


















