# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: nill	%r0, 0                  # encoding: [0xa5,0x07,0x00,0x00]
#CHECK: nill	%r0, 32768              # encoding: [0xa5,0x07,0x80,0x00]
#CHECK: nill	%r0, 65535              # encoding: [0xa5,0x07,0xff,0xff]
#CHECK: nill	%r15, 0                 # encoding: [0xa5,0xf7,0x00,0x00]

	nill	%r0, 0
	nill	%r0, 0x8000
	nill	%r0, 0xffff
	nill	%r15, 0
