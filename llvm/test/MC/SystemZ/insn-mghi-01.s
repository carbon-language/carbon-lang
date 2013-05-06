# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: mghi	%r0, -32768             # encoding: [0xa7,0x0d,0x80,0x00]
#CHECK: mghi	%r0, -1                 # encoding: [0xa7,0x0d,0xff,0xff]
#CHECK: mghi	%r0, 0                  # encoding: [0xa7,0x0d,0x00,0x00]
#CHECK: mghi	%r0, 1                  # encoding: [0xa7,0x0d,0x00,0x01]
#CHECK: mghi	%r0, 32767              # encoding: [0xa7,0x0d,0x7f,0xff]
#CHECK: mghi	%r15, 0                 # encoding: [0xa7,0xfd,0x00,0x00]

	mghi	%r0, -32768
	mghi	%r0, -1
	mghi	%r0, 0
	mghi	%r0, 1
	mghi	%r0, 32767
	mghi	%r15, 0
