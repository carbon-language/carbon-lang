# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: slgr	%r0, %r0                # encoding: [0xb9,0x0b,0x00,0x00]
#CHECK: slgr	%r0, %r15               # encoding: [0xb9,0x0b,0x00,0x0f]
#CHECK: slgr	%r15, %r0               # encoding: [0xb9,0x0b,0x00,0xf0]
#CHECK: slgr	%r7, %r8                # encoding: [0xb9,0x0b,0x00,0x78]

	slgr	%r0,%r0
	slgr	%r0,%r15
	slgr	%r15,%r0
	slgr	%r7,%r8
