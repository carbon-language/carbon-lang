# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ar	%r0, %r0                # encoding: [0x1a,0x00]
#CHECK: ar	%r0, %r15               # encoding: [0x1a,0x0f]
#CHECK: ar	%r15, %r0               # encoding: [0x1a,0xf0]
#CHECK: ar	%r7, %r8                # encoding: [0x1a,0x78]

	ar	%r0,%r0
	ar	%r0,%r15
	ar	%r15,%r0
	ar	%r7,%r8
