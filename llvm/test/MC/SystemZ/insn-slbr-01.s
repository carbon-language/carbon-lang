# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: slbr	%r0, %r0                # encoding: [0xb9,0x99,0x00,0x00]
#CHECK: slbr	%r0, %r15               # encoding: [0xb9,0x99,0x00,0x0f]
#CHECK: slbr	%r15, %r0               # encoding: [0xb9,0x99,0x00,0xf0]
#CHECK: slbr	%r7, %r8                # encoding: [0xb9,0x99,0x00,0x78]

	slbr	%r0,%r0
	slbr	%r0,%r15
	slbr	%r15,%r0
	slbr	%r7,%r8
