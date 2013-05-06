# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lbr	%r0, %r15               # encoding: [0xb9,0x26,0x00,0x0f]
#CHECK: lbr	%r7, %r8                # encoding: [0xb9,0x26,0x00,0x78]
#CHECK: lbr	%r15, %r0               # encoding: [0xb9,0x26,0x00,0xf0]

	lbr	%r0, %r15
	lbr	%r7, %r8
	lbr	%r15, %r0
