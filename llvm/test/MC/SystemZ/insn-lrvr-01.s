# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lrvr	%r0, %r0                # encoding: [0xb9,0x1f,0x00,0x00]
#CHECK: lrvr	%r0, %r15               # encoding: [0xb9,0x1f,0x00,0x0f]
#CHECK: lrvr	%r15, %r0               # encoding: [0xb9,0x1f,0x00,0xf0]
#CHECK: lrvr	%r7, %r8                # encoding: [0xb9,0x1f,0x00,0x78]
#CHECK: lrvr	%r15, %r15              # encoding: [0xb9,0x1f,0x00,0xff]

	lrvr	%r0,%r0
	lrvr	%r0,%r15
	lrvr	%r15,%r0
	lrvr	%r7,%r8
	lrvr	%r15,%r15
