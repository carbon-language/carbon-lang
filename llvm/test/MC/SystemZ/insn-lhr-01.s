# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lhr	%r0, %r15               # encoding: [0xb9,0x27,0x00,0x0f]
#CHECK: lhr	%r7, %r8                # encoding: [0xb9,0x27,0x00,0x78]
#CHECK: lhr	%r15, %r0               # encoding: [0xb9,0x27,0x00,0xf0]

	lhr	%r0, %r15
	lhr	%r7, %r8
	lhr	%r15, %r0
