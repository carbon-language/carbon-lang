# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lgbr	%r0, %r15               # encoding: [0xb9,0x06,0x00,0x0f]
#CHECK: lgbr	%r7, %r8                # encoding: [0xb9,0x06,0x00,0x78]
#CHECK: lgbr	%r15, %r0               # encoding: [0xb9,0x06,0x00,0xf0]

	lgbr	%r0, %r15
	lgbr	%r7, %r8
	lgbr	%r15, %r0
