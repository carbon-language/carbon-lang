# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: iill	%r0, 0                  # encoding: [0xa5,0x03,0x00,0x00]
#CHECK: iill	%r0, 32768              # encoding: [0xa5,0x03,0x80,0x00]
#CHECK: iill	%r0, 65535              # encoding: [0xa5,0x03,0xff,0xff]
#CHECK: iill	%r15, 0                 # encoding: [0xa5,0xf3,0x00,0x00]

	iill	%r0, 0
	iill	%r0, 0x8000
	iill	%r0, 0xffff
	iill	%r15, 0
