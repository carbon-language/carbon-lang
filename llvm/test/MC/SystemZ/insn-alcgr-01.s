# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: alcgr	%r0, %r0                # encoding: [0xb9,0x88,0x00,0x00]
#CHECK: alcgr	%r0, %r15               # encoding: [0xb9,0x88,0x00,0x0f]
#CHECK: alcgr	%r15, %r0               # encoding: [0xb9,0x88,0x00,0xf0]
#CHECK: alcgr	%r7, %r8                # encoding: [0xb9,0x88,0x00,0x78]

	alcgr	%r0,%r0
	alcgr	%r0,%r15
	alcgr	%r15,%r0
	alcgr	%r7,%r8
