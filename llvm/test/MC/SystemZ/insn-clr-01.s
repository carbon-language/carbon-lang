# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clr	%r0, %r0                # encoding: [0x15,0x00]
#CHECK: clr	%r0, %r15               # encoding: [0x15,0x0f]
#CHECK: clr	%r15, %r0               # encoding: [0x15,0xf0]
#CHECK: clr	%r7, %r8                # encoding: [0x15,0x78]

	clr	%r0,%r0
	clr	%r0,%r15
	clr	%r15,%r0
	clr	%r7,%r8
