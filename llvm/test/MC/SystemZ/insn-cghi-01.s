# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cghi	%r0, -32768             # encoding: [0xa7,0x0f,0x80,0x00]
#CHECK: cghi	%r0, -1                 # encoding: [0xa7,0x0f,0xff,0xff]
#CHECK: cghi	%r0, 0                  # encoding: [0xa7,0x0f,0x00,0x00]
#CHECK: cghi	%r0, 1                  # encoding: [0xa7,0x0f,0x00,0x01]
#CHECK: cghi	%r0, 32767              # encoding: [0xa7,0x0f,0x7f,0xff]
#CHECK: cghi	%r15, 0                 # encoding: [0xa7,0xff,0x00,0x00]

	cghi	%r0, -32768
	cghi	%r0, -1
	cghi	%r0, 0
	cghi	%r0, 1
	cghi	%r0, 32767
	cghi	%r15, 0
