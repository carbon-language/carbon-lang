#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

ifndef omp_os

  # Windows sets environment variable OS; for other systems, ask uname
  ifeq ($(OS),)
    OS:=$(shell uname)
    ifeq ($(OS),)
      $(error "Cannot detect operating system")
    endif
    export omp_os=$(OS)
  endif

  ifeq ($(OS), Windows_NT)
    export omp_os=windows
  endif
  ifeq ($(OS), Linux)
    export omp_os=linux
  endif
  ifeq ($(OS), FreeBSD)
    export omp_os=freebsd
  endif
  ifeq ($(OS), Darwin)
    export omp_os=macos
  endif

endif # !omp_os

# Compiling for the Intel(R) Many Integrated Core architecture is non-trivial at the next layer
# of script down, but we can make it consistent here.
ifeq "$(arch)" "mic"
    # I really do mean this...
    override arch:=32e
    override mic:=yes
else
    override mic:=no
endif

ifeq (,$(wildcard $(omp_root)/tools/$(omp_os).inc))
  $(error "$(omp_os)" is not supported. Add tools/$(omp_os).inc file with os-specific settings )
endif

# detect arch and runtime versions, provide common host-specific definitions
include $(omp_root)/tools/$(omp_os).inc

ifeq ($(arch),)
 $(error Architecture not detected)
endif

# Setting defaults
mode?=release

ifeq "$(filter 32 32e 64,$(arch))" ""
    compiler?=gcc
else
    ifeq "$(omp_os)" "windows"
        compiler?=icl
    else
        compiler?=icc
    endif
endif

ifneq "$(mic)" "no"
    ifeq "$(compiler)" "gcc"
	$(error Compiling the runtime with gcc is not supported on Intel\(R\) Many Integrated Core Architecture)
    endif

    # Magic flags for the build script!
    build_args += --os=lrb --mic-arch=knc --mic-os=lin --mic-comp=offload

    # Check that the binutils for Intel(R) Many Integrated Core Architecture are available
    # First we see whether the objdump on the user's path supports the k1om architecture.
    hask1om = $(shell if (objdump --help | grep -s k1om); then echo OK; else echo KO; fi)
    ifneq "$(hask1om)" "OK"
        # Appropriate binutils are not already set up, so try to add them from the default place.
        micBinPath = /usr/linux-k1om-4.7/x86_64-k1om-linux/bin
        micBinPresent = $(shell if test -d $(micBinPath); then echo OK; else echo KO; fi)
        ifneq "$(micBinPresent)" "OK"
            # We can't find them in the normal place, so complain.
            $(error Compiling for Intel\(R\) Many Integrated Core Architecture requires that the cross-hosted binutils are available in $(micBinPath).\
See the Tools tab at http://software.intel.com/mic-developer)
        endif
        export PATH := $(micBinPath):${PATH}
    endif
endif

export BUILD_COMPILER := $(compiler)


