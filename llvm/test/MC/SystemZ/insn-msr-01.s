# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: msr	%r0, %r0                # encoding: [0xb2,0x52,0x00,0x00]
#CHECK: msr	%r0, %r15               # encoding: [0xb2,0x52,0x00,0x0f]
#CHECK: msr	%r15, %r0               # encoding: [0xb2,0x52,0x00,0xf0]
#CHECK: msr	%r7, %r8                # encoding: [0xb2,0x52,0x00,0x78]

	msr	%r0,%r0
	msr	%r0,%r15
	msr	%r15,%r0
	msr	%r7,%r8
