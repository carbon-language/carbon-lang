# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lcgr	%r0, %r0                # encoding: [0xb9,0x03,0x00,0x00]
#CHECK: lcgr	%r0, %r15               # encoding: [0xb9,0x03,0x00,0x0f]
#CHECK: lcgr	%r15, %r0               # encoding: [0xb9,0x03,0x00,0xf0]
#CHECK: lcgr	%r7, %r8                # encoding: [0xb9,0x03,0x00,0x78]

	lcgr	%r0,%r0
	lcgr	%r0,%r15
	lcgr	%r15,%r0
	lcgr	%r7,%r8
