# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llhr	%r0, %r15               # encoding: [0xb9,0x95,0x00,0x0f]
#CHECK: llhr	%r7, %r8                # encoding: [0xb9,0x95,0x00,0x78]
#CHECK: llhr	%r15, %r0               # encoding: [0xb9,0x95,0x00,0xf0]

	llhr	%r0, %r15
	llhr	%r7, %r8
	llhr	%r15, %r0
