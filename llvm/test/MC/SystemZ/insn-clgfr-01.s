# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clgfr	%r0, %r0                # encoding: [0xb9,0x31,0x00,0x00]
#CHECK: clgfr	%r0, %r15               # encoding: [0xb9,0x31,0x00,0x0f]
#CHECK: clgfr	%r15, %r0               # encoding: [0xb9,0x31,0x00,0xf0]
#CHECK: clgfr	%r7, %r8                # encoding: [0xb9,0x31,0x00,0x78]

	clgfr	%r0,%r0
	clgfr	%r0,%r15
	clgfr	%r15,%r0
	clgfr	%r7,%r8
