# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sr	%r0, %r0                # encoding: [0x1b,0x00]
#CHECK: sr	%r0, %r15               # encoding: [0x1b,0x0f]
#CHECK: sr	%r15, %r0               # encoding: [0x1b,0xf0]
#CHECK: sr	%r7, %r8                # encoding: [0x1b,0x78]

	sr	%r0,%r0
	sr	%r0,%r15
	sr	%r15,%r0
	sr	%r7,%r8
