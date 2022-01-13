Testing LLDB using QEMU
=======================

.. contents::
   :local:

QEMU system mode emulation
--------------------------

QEMU can be used to test LLDB in an emulation environment in the absence of
actual hardware. This page describes instructions to help setup a QEMU emulation
environment for testing LLDB.

The scripts under llvm-project/lldb/scripts/lldb-test-qemu can quickly help
setup a virtual LLDB testing environment using QEMU. The scripts currently work
with Arm or AArch64, but support for other architectures can be added easily.

* **setup.sh** is used to build the Linux kernel image and QEMU system emulation executable(s) from source.
* **rootfs.sh** is used to generate Ubuntu root file system images to be used for QEMU system mode emulation.
* **run-qemu.sh** utilizes QEMU to boot a Linux kernel image with a root file system image.

Once we have booted our kernel we can run lldb-server in emulation environment.
Ubuntu Bionic/Focal x86_64 host was used to test these scripts instructions in this
document. Please update it according to your host distribution/architecture.

.. note::
  Instructions on this page and QEMU helper scripts are verified on a Ubuntu Bionic/Focal (x86_64) host. Moreover, scripts require sudo/root permissions for installing dependencies and setting up QEMU host/guest network.

Given below are some examples of common use-cases of LLDB QEMU testing
helper scripts:

Create Ubuntu root file system image for QEMU system emulation with rootfs.sh
--------------------------------------------------------------------------------

**Example:** generate Ubuntu Bionic (armhf) rootfs image of size 1 GB
::

  $ bash rootfs.sh --arch armhf --distro bionic --size 1G

**Example:** generate Ubuntu Focal (arm64) rootfs image of size 2 GB
::

  $ bash rootfs.sh --arch arm64 --distro focal --size 2G

rootfs.sh has been tested for generating Ubuntu Bionic and Focal images but they can be used to generate rootfs images of other Debian Linux distribution.

rootfs.sh defaults username of generated image to your current username on host computer.


Build QEMU or cross compile Linux kernel from source using setup.sh
-----------------------------------------------------------------------

**Example:** Build QEMU binaries and Arm/AArch64 Linux kernel image
::

$ bash setup.sh --qemu --kernel arm
$ bash setup.sh --qemu --kernel arm64

**Example:** Build Linux kernel image only
::

$ bash setup.sh --kernel arm
$ bash setup.sh --kernel arm64

**Example:** Build qemu-system-arm and qemu-system-aarch64 binaries.
::

$ bash setup.sh --qemu

**Example:** Remove qemu.git, linux.git and linux.build from working directory
::

$ bash setup.sh --clean


Run QEMU Arm or AArch64 system emulation using run-qemu.sh
----------------------------------------------------------
run-qemu.sh has following dependencies:

* Follow https://wiki.qemu.org/Documentation/Networking/NAT and set up bridge
  networking for QEMU.

* Make sure /etc/qemu-ifup script is available with executable permissions.

* QEMU binaries must be built from source using setup.sh or provided via --qemu
  commandline argument.

* Linux kernel image must be built from source using setup.sh or provided via
  --kernel commandline argument.

* linux.build and qemu.git folder must be present in current directory if
  setup.sh was used to build Linux kernel and QEMU binaries.

* --sve option will enable AArch64 SVE mode.

* --mte option will enable AArch64 MTE (memory tagging) mode.
  (can be used on its own or in addition to --sve)


**Example:** Run QEMU Arm or AArch64 system emulation using run-qemu.sh
::

  $ sudo bash run-qemu.sh --arch arm --rootfs <path of rootfs image>
  $ sudo bash run-qemu.sh --arch arm64 --rootfs <path of rootfs image>

**Example:** Run QEMU with kernel image and qemu binary provided using commandline
::

  $ sudo bash run-qemu.sh --arch arm64 --rootfs <path of rootfs image> \
  --kernel <path of Linux kernel image> --qemu <path of QEMU binary>


Steps for running lldb-server in QEMU system emulation environment
------------------------------------------------------------------

* Make sure bridge networking is enabled between host machine and QEMU VM

* Find out ip address assigned to eth0 in emulation environment

* Setup ssh access between host machine and emulation environment

* Login emulation environment and install dependencies

::

  $ sudo apt install python-dev libedit-dev libncurses5-dev libexpat1-dev

* Cross compile LLDB server for AArch64 Linux: Please visit https://lldb.llvm.org/resources/build.html for instructions on how to cross compile LLDB server.

* Transfer LLDB server executable to emulation environment

::

  $ scp lldb-server username@ip-address-of-emulation-environment:/home/username

* Run lldb-server inside QEMU VM

* Try connecting to lldb-server running inside QEMU VM with selected ip:port
