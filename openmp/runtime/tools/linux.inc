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

####### Detections and Commands ###############################################

ifndef arch
        uname_m:=$(shell uname -m)
        ifeq ($(uname_m),i686)
                export arch:=32
        endif
        ifeq ($(uname_m),x86_64)
                export arch:=32e
        endif
        ifndef arch
                export arch:=$(uname_m)
        endif
endif

CMD=sh -c
CWD=$(shell pwd)
CP=cp
RM?=rm -f
RMR?=rm -rf
RD?=rmdir
MD?=mkdir -p
NUL= /dev/null
SLASH=/
