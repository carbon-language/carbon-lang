# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: nihl	%r0, 0                  # encoding: [0xa5,0x05,0x00,0x00]
#CHECK: nihl	%r0, 32768              # encoding: [0xa5,0x05,0x80,0x00]
#CHECK: nihl	%r0, 65535              # encoding: [0xa5,0x05,0xff,0xff]
#CHECK: nihl	%r15, 0                 # encoding: [0xa5,0xf5,0x00,0x00]

	nihl	%r0, 0
	nihl	%r0, 0x8000
	nihl	%r0, 0xffff
	nihl	%r15, 0
