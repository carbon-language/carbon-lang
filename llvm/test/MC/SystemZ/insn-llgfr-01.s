# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llgfr	%r0, %r15               # encoding: [0xb9,0x16,0x00,0x0f]
#CHECK: llgfr	%r7, %r8                # encoding: [0xb9,0x16,0x00,0x78]
#CHECK: llgfr	%r15, %r0               # encoding: [0xb9,0x16,0x00,0xf0]

	llgfr	%r0, %r15
	llgfr	%r7, %r8
	llgfr	%r15, %r0
