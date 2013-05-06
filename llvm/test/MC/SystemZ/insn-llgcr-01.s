# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llgcr	%r0, %r15               # encoding: [0xb9,0x84,0x00,0x0f]
#CHECK: llgcr	%r7, %r8                # encoding: [0xb9,0x84,0x00,0x78]
#CHECK: llgcr	%r15, %r0               # encoding: [0xb9,0x84,0x00,0xf0]

	llgcr	%r0, %r15
	llgcr	%r7, %r8
	llgcr	%r15, %r0
