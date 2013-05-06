# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: iihl	%r0, 0                  # encoding: [0xa5,0x01,0x00,0x00]
#CHECK: iihl	%r0, 32768              # encoding: [0xa5,0x01,0x80,0x00]
#CHECK: iihl	%r0, 65535              # encoding: [0xa5,0x01,0xff,0xff]
#CHECK: iihl	%r15, 0                 # encoding: [0xa5,0xf1,0x00,0x00]

	iihl	%r0, 0
	iihl	%r0, 0x8000
	iihl	%r0, 0xffff
	iihl	%r15, 0
