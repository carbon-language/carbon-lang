# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llill	%r0, 0                  # encoding: [0xa5,0x0f,0x00,0x00]
#CHECK: llill	%r0, 32768              # encoding: [0xa5,0x0f,0x80,0x00]
#CHECK: llill	%r0, 65535              # encoding: [0xa5,0x0f,0xff,0xff]
#CHECK: llill	%r15, 0                 # encoding: [0xa5,0xff,0x00,0x00]

	llill	%r0, 0
	llill	%r0, 0x8000
	llill	%r0, 0xffff
	llill	%r15, 0
