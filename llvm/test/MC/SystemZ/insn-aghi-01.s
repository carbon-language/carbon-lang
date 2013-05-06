# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: aghi	%r0, -32768             # encoding: [0xa7,0x0b,0x80,0x00]
#CHECK: aghi	%r0, -1                 # encoding: [0xa7,0x0b,0xff,0xff]
#CHECK: aghi	%r0, 0                  # encoding: [0xa7,0x0b,0x00,0x00]
#CHECK: aghi	%r0, 1                  # encoding: [0xa7,0x0b,0x00,0x01]
#CHECK: aghi	%r0, 32767              # encoding: [0xa7,0x0b,0x7f,0xff]
#CHECK: aghi	%r15, 0                 # encoding: [0xa7,0xfb,0x00,0x00]

	aghi	%r0, -32768
	aghi	%r0, -1
	aghi	%r0, 0
	aghi	%r0, 1
	aghi	%r0, 32767
	aghi	%r15, 0
