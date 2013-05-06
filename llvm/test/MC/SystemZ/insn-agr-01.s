# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: agr	%r0, %r0                # encoding: [0xb9,0x08,0x00,0x00]
#CHECK: agr	%r0, %r15               # encoding: [0xb9,0x08,0x00,0x0f]
#CHECK: agr	%r15, %r0               # encoding: [0xb9,0x08,0x00,0xf0]
#CHECK: agr	%r7, %r8                # encoding: [0xb9,0x08,0x00,0x78]

	agr	%r0,%r0
	agr	%r0,%r15
	agr	%r15,%r0
	agr	%r7,%r8
