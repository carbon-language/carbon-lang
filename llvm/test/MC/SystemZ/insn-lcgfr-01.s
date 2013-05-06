# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lcgfr	%r0, %r0                # encoding: [0xb9,0x13,0x00,0x00]
#CHECK: lcgfr	%r0, %r15               # encoding: [0xb9,0x13,0x00,0x0f]
#CHECK: lcgfr	%r15, %r0               # encoding: [0xb9,0x13,0x00,0xf0]
#CHECK: lcgfr	%r7, %r8                # encoding: [0xb9,0x13,0x00,0x78]

	lcgfr	%r0,%r0
	lcgfr	%r0,%r15
	lcgfr	%r15,%r0
	lcgfr	%r7,%r8
