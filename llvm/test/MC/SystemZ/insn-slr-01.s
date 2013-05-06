# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: slr	%r0, %r0                # encoding: [0x1f,0x00]
#CHECK: slr	%r0, %r15               # encoding: [0x1f,0x0f]
#CHECK: slr	%r15, %r0               # encoding: [0x1f,0xf0]
#CHECK: slr	%r7, %r8                # encoding: [0x1f,0x78]

	slr	%r0,%r0
	slr	%r0,%r15
	slr	%r15,%r0
	slr	%r7,%r8
