# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lgfr	%r0, %r15               # encoding: [0xb9,0x14,0x00,0x0f]
#CHECK: lgfr	%r7, %r8                # encoding: [0xb9,0x14,0x00,0x78]
#CHECK: lgfr	%r15, %r0               # encoding: [0xb9,0x14,0x00,0xf0]

	lgfr	%r0, %r15
	lgfr	%r7, %r8
	lgfr	%r15, %r0
