# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lrvgr	%r0, %r0                # encoding: [0xb9,0x0f,0x00,0x00]
#CHECK: lrvgr	%r0, %r15               # encoding: [0xb9,0x0f,0x00,0x0f]
#CHECK: lrvgr	%r15, %r0               # encoding: [0xb9,0x0f,0x00,0xf0]
#CHECK: lrvgr	%r7, %r8                # encoding: [0xb9,0x0f,0x00,0x78]
#CHECK: lrvgr	%r15, %r15              # encoding: [0xb9,0x0f,0x00,0xff]

	lrvgr	%r0,%r0
	lrvgr	%r0,%r15
	lrvgr	%r15,%r0
	lrvgr	%r7,%r8
	lrvgr	%r15,%r15
