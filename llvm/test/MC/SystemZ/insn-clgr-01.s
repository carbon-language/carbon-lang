# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clgr	%r0, %r0                # encoding: [0xb9,0x21,0x00,0x00]
#CHECK: clgr	%r0, %r15               # encoding: [0xb9,0x21,0x00,0x0f]
#CHECK: clgr	%r15, %r0               # encoding: [0xb9,0x21,0x00,0xf0]
#CHECK: clgr	%r7, %r8                # encoding: [0xb9,0x21,0x00,0x78]

	clgr	%r0,%r0
	clgr	%r0,%r15
	clgr	%r15,%r0
	clgr	%r7,%r8
