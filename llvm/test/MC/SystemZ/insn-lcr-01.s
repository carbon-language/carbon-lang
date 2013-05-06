# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lcr	%r0, %r0                # encoding: [0x13,0x00]
#CHECK: lcr	%r0, %r15               # encoding: [0x13,0x0f]
#CHECK: lcr	%r15, %r0               # encoding: [0x13,0xf0]
#CHECK: lcr	%r7, %r8                # encoding: [0x13,0x78]

	lcr	%r0,%r0
	lcr	%r0,%r15
	lcr	%r15,%r0
	lcr	%r7,%r8
