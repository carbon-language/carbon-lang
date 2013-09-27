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
 ifeq ($(shell /usr/sbin/sysctl -n hw.machine),Power Macintosh)
   ifeq ($(shell /usr/sbin/sysctl -n hw.optional.64bitops),1)
     export arch:=ppc64
   else
     export arch:=ppc32
   endif
 else
   ifeq ($(shell /usr/sbin/sysctl -n hw.optional.x86_64 2>/dev/null),1)
     export arch:=intel64
   else
     export arch:=ia32
   endif
 endif
endif

CMD=$(SHELL) -c
CWD=$(shell pwd)
RM?=rm -f
RMR?=rm -rf
RD?=rmdir
MD?=mkdir -p
NUL= /dev/null
SLASH=/
