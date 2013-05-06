# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: oill	%r0, 0                  # encoding: [0xa5,0x0b,0x00,0x00]
#CHECK: oill	%r0, 32768              # encoding: [0xa5,0x0b,0x80,0x00]
#CHECK: oill	%r0, 65535              # encoding: [0xa5,0x0b,0xff,0xff]
#CHECK: oill	%r15, 0                 # encoding: [0xa5,0xfb,0x00,0x00]

	oill	%r0, 0
	oill	%r0, 0x8000
	oill	%r0, 0xffff
	oill	%r15, 0
