# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: oihl	%r0, 0                  # encoding: [0xa5,0x09,0x00,0x00]
#CHECK: oihl	%r0, 32768              # encoding: [0xa5,0x09,0x80,0x00]
#CHECK: oihl	%r0, 65535              # encoding: [0xa5,0x09,0xff,0xff]
#CHECK: oihl	%r15, 0                 # encoding: [0xa5,0xf9,0x00,0x00]

	oihl	%r0, 0
	oihl	%r0, 0x8000
	oihl	%r0, 0xffff
	oihl	%r15, 0
