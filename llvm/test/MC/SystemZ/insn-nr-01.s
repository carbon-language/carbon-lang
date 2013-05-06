# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: nr	%r0, %r0                # encoding: [0x14,0x00]
#CHECK: nr	%r0, %r15               # encoding: [0x14,0x0f]
#CHECK: nr	%r15, %r0               # encoding: [0x14,0xf0]
#CHECK: nr	%r7, %r8                # encoding: [0x14,0x78]

	nr	%r0,%r0
	nr	%r0,%r15
	nr	%r15,%r0
	nr	%r7,%r8
