# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: algr	%r0, %r0                # encoding: [0xb9,0x0a,0x00,0x00]
#CHECK: algr	%r0, %r15               # encoding: [0xb9,0x0a,0x00,0x0f]
#CHECK: algr	%r15, %r0               # encoding: [0xb9,0x0a,0x00,0xf0]
#CHECK: algr	%r7, %r8                # encoding: [0xb9,0x0a,0x00,0x78]

	algr	%r0,%r0
	algr	%r0,%r15
	algr	%r15,%r0
	algr	%r7,%r8
