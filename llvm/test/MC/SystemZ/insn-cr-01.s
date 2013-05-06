# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cr	%r0, %r0                # encoding: [0x19,0x00]
#CHECK: cr	%r0, %r15               # encoding: [0x19,0x0f]
#CHECK: cr	%r15, %r0               # encoding: [0x19,0xf0]
#CHECK: cr	%r7, %r8                # encoding: [0x19,0x78]

	cr	%r0,%r0
	cr	%r0,%r15
	cr	%r15,%r0
	cr	%r7,%r8
