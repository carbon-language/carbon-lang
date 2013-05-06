# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lghr	%r0, %r15               # encoding: [0xb9,0x07,0x00,0x0f]
#CHECK: lghr	%r7, %r8                # encoding: [0xb9,0x07,0x00,0x78]
#CHECK: lghr	%r15, %r0               # encoding: [0xb9,0x07,0x00,0xf0]

	lghr	%r0, %r15
	lghr	%r7, %r8
	lghr	%r15, %r0
