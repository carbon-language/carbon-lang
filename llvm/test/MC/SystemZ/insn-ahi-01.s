# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ahi	%r0, -32768             # encoding: [0xa7,0x0a,0x80,0x00]
#CHECK: ahi	%r0, -1                 # encoding: [0xa7,0x0a,0xff,0xff]
#CHECK: ahi	%r0, 0                  # encoding: [0xa7,0x0a,0x00,0x00]
#CHECK: ahi	%r0, 1                  # encoding: [0xa7,0x0a,0x00,0x01]
#CHECK: ahi	%r0, 32767              # encoding: [0xa7,0x0a,0x7f,0xff]
#CHECK: ahi	%r15, 0                 # encoding: [0xa7,0xfa,0x00,0x00]

	ahi	%r0, -32768
	ahi	%r0, -1
	ahi	%r0, 0
	ahi	%r0, 1
	ahi	%r0, 32767
	ahi	%r15, 0
