# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: chi	%r0, -32768             # encoding: [0xa7,0x0e,0x80,0x00]
#CHECK: chi	%r0, -1                 # encoding: [0xa7,0x0e,0xff,0xff]
#CHECK: chi	%r0, 0                  # encoding: [0xa7,0x0e,0x00,0x00]
#CHECK: chi	%r0, 1                  # encoding: [0xa7,0x0e,0x00,0x01]
#CHECK: chi	%r0, 32767              # encoding: [0xa7,0x0e,0x7f,0xff]
#CHECK: chi	%r15, 0                 # encoding: [0xa7,0xfe,0x00,0x00]

	chi	%r0, -32768
	chi	%r0, -1
	chi	%r0, 0
	chi	%r0, 1
	chi	%r0, 32767
	chi	%r15, 0
