# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: agfr	%r0, %r0                # encoding: [0xb9,0x18,0x00,0x00]
#CHECK: agfr	%r0, %r15               # encoding: [0xb9,0x18,0x00,0x0f]
#CHECK: agfr	%r15, %r0               # encoding: [0xb9,0x18,0x00,0xf0]
#CHECK: agfr	%r7, %r8                # encoding: [0xb9,0x18,0x00,0x78]

	agfr	%r0,%r0
	agfr	%r0,%r15
	agfr	%r15,%r0
	agfr	%r7,%r8
