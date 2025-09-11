# Panda Franka Robot Device
First check whether current operation system is under real-time kernel by `uname -a`, refer to <https://frankaemika.github.io/docs/installation_linux.html#verifying-the-new-kernel>. If not, refer to <https://frankaemika.github.io/docs/installation_linux.html#setting-up-the-real-time-kernel> to set up it.
Then you can first try running `python r3kit/robot/franka/panda.py`. If it works, nothing to do more. Otherwise, it may throw exception called the mismatch between `libfranka` and `frankx`. Refer to <https://frankaemika.github.io/docs/libfranka_changelog.html> and <https://frankaemika.github.io/docs/installation_linux.html> to build and install the other version of `libfranka`. After build `libfranka`, refer to <https://github.com/pantor/frankx#installation> to build and install `frankx`.

```bash
# install dependencies first
sudo apt install build-essential cmake git libpoco-dev libeigen3-dev
# build libfranka from source
git clone --recursive https://github.com/frankaemika/libfranka
cd libfranka
git checkout 0.9.2
git submodule update
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
cmake --build .
cpack -G DEB
sudo dpkg -i libfranka*.deb
```
```bash
# install pybind11 first
pip install "pybind11[global]"
# build frankx from source
git clone --recurse-submodules git@github.com:pantor/frankx.git
cd frankx
git checkout v0.2.0
mkdir -p build
cd build
cmake -DBUILD_TYPE=Release .. -DBUILD_TESTS=OFF
make
sudo make install
cd ..
pip install .
```
