# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: slgfr	%r0, %r0                # encoding: [0xb9,0x1b,0x00,0x00]
#CHECK: slgfr	%r0, %r15               # encoding: [0xb9,0x1b,0x00,0x0f]
#CHECK: slgfr	%r15, %r0               # encoding: [0xb9,0x1b,0x00,0xf0]
#CHECK: slgfr	%r7, %r8                # encoding: [0xb9,0x1b,0x00,0x78]

	slgfr	%r0,%r0
	slgfr	%r0,%r15
	slgfr	%r15,%r0
	slgfr	%r7,%r8
