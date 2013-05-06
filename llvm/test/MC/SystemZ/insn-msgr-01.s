# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: msgr	%r0, %r0                # encoding: [0xb9,0x0c,0x00,0x00]
#CHECK: msgr	%r0, %r15               # encoding: [0xb9,0x0c,0x00,0x0f]
#CHECK: msgr	%r15, %r0               # encoding: [0xb9,0x0c,0x00,0xf0]
#CHECK: msgr	%r7, %r8                # encoding: [0xb9,0x0c,0x00,0x78]

	msgr	%r0,%r0
	msgr	%r0,%r15
	msgr	%r15,%r0
	msgr	%r7,%r8
