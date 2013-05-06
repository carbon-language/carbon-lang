# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llilh	%r0, 0                  # encoding: [0xa5,0x0e,0x00,0x00]
#CHECK: llilh	%r0, 32768              # encoding: [0xa5,0x0e,0x80,0x00]
#CHECK: llilh	%r0, 65535              # encoding: [0xa5,0x0e,0xff,0xff]
#CHECK: llilh	%r15, 0                 # encoding: [0xa5,0xfe,0x00,0x00]

	llilh	%r0, 0
	llilh	%r0, 0x8000
	llilh	%r0, 0xffff
	llilh	%r15, 0
