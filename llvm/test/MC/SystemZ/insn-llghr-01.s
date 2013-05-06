# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llghr	%r0, %r15               # encoding: [0xb9,0x85,0x00,0x0f]
#CHECK: llghr	%r7, %r8                # encoding: [0xb9,0x85,0x00,0x78]
#CHECK: llghr	%r15, %r0               # encoding: [0xb9,0x85,0x00,0xf0]

	llghr	%r0, %r15
	llghr	%r7, %r8
	llghr	%r15, %r0
