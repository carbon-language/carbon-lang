# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lghi	%r0, -32768             # encoding: [0xa7,0x09,0x80,0x00]
#CHECK: lghi	%r0, -1                 # encoding: [0xa7,0x09,0xff,0xff]
#CHECK: lghi	%r0, 0                  # encoding: [0xa7,0x09,0x00,0x00]
#CHECK: lghi	%r0, 1                  # encoding: [0xa7,0x09,0x00,0x01]
#CHECK: lghi	%r0, 32767              # encoding: [0xa7,0x09,0x7f,0xff]
#CHECK: lghi	%r15, 0                 # encoding: [0xa7,0xf9,0x00,0x00]

	lghi	%r0, -32768
	lghi	%r0, -1
	lghi	%r0, 0
	lghi	%r0, 1
	lghi	%r0, 32767
	lghi	%r15, 0
