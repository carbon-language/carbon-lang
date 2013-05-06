# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llcr	%r0, %r15               # encoding: [0xb9,0x94,0x00,0x0f]
#CHECK: llcr	%r7, %r8                # encoding: [0xb9,0x94,0x00,0x78]
#CHECK: llcr	%r15, %r0               # encoding: [0xb9,0x94,0x00,0xf0]

	llcr	%r0, %r15
	llcr	%r7, %r8
	llcr	%r15, %r0
