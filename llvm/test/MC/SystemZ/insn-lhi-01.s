# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lhi	%r0, -32768             # encoding: [0xa7,0x08,0x80,0x00]
#CHECK: lhi	%r0, -1                 # encoding: [0xa7,0x08,0xff,0xff]
#CHECK: lhi	%r0, 0                  # encoding: [0xa7,0x08,0x00,0x00]
#CHECK: lhi	%r0, 1                  # encoding: [0xa7,0x08,0x00,0x01]
#CHECK: lhi	%r0, 32767              # encoding: [0xa7,0x08,0x7f,0xff]
#CHECK: lhi	%r15, 0                 # encoding: [0xa7,0xf8,0x00,0x00]

	lhi	%r0, -32768
	lhi	%r0, -1
	lhi	%r0, 0
	lhi	%r0, 1
	lhi	%r0, 32767
	lhi	%r15, 0
