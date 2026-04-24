# GPU-Accelerated SMR



Link for cuda installation guide: [Cuda Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

# Summarizing the instructions for Ubuntu 24.04 LTS:
* Supported Linux Distributions: *distro=ubuntu2204*, *arch=amd64*

* Native Linux Distribution Support and Validated OS Versions for CUDA 13.2 Update 1: *OS version=24.04.4*, *Kernel=6.17.0-19*, *Default GCC=14.3.0*, *GLIBC=2.39*
## 1. System Requirements
To use NVIDIA CUDA on your system, you will need the following installed:
* CUDA-capable GPU

* A supported version of Linux with a gcc compiler and toolchain

* CUDA Toolkit (available at [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads))

## 2. Pre-installation Actions
**Verify the system has a CUDA-capable GPU:** 
```bash 
lspci | grep -i nvidia
```

**Verify the system is running a supported version of Linux:**
```bash
hostnamectl
```

**Verify the system has gcc installed:**
```bash
gcc --version
```

Download the NVIDIA CUDA Toolkit: The NVIDIA CUDA Toolkit is available at [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

**Handle conflicting installation methods:** Before installing CUDA, any previous installations that could conflict should be uninstalled. This will not affect systems which have not had CUDA installed previously, or systems where the installation method has been preserved (RPM/Deb vs. Runfile).

## 3. Package Manager Installation
### Network Repository Installation
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/amd64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
```

### Common Installation Instructions
```bash
apt update
apt install cuda-toolkit
apt install nvidia-gds
reboot
```

## Post-installation Action
### Mandatory Actions