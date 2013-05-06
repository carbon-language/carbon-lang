# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ngr	%r0, %r0                # encoding: [0xb9,0x80,0x00,0x00]
#CHECK: ngr	%r0, %r15               # encoding: [0xb9,0x80,0x00,0x0f]
#CHECK: ngr	%r15, %r0               # encoding: [0xb9,0x80,0x00,0xf0]
#CHECK: ngr	%r7, %r8                # encoding: [0xb9,0x80,0x00,0x78]

	ngr	%r0,%r0
	ngr	%r0,%r15
	ngr	%r15,%r0
	ngr	%r7,%r8
