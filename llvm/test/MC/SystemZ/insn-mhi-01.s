# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: mhi	%r0, -32768             # encoding: [0xa7,0x0c,0x80,0x00]
#CHECK: mhi	%r0, -1                 # encoding: [0xa7,0x0c,0xff,0xff]
#CHECK: mhi	%r0, 0                  # encoding: [0xa7,0x0c,0x00,0x00]
#CHECK: mhi	%r0, 1                  # encoding: [0xa7,0x0c,0x00,0x01]
#CHECK: mhi	%r0, 32767              # encoding: [0xa7,0x0c,0x7f,0xff]
#CHECK: mhi	%r15, 0                 # encoding: [0xa7,0xfc,0x00,0x00]

	mhi	%r0, -32768
	mhi	%r0, -1
	mhi	%r0, 0
	mhi	%r0, 1
	mhi	%r0, 32767
	mhi	%r15, 0
