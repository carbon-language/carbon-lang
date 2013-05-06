# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: slbgr	%r0, %r0                # encoding: [0xb9,0x89,0x00,0x00]
#CHECK: slbgr	%r0, %r15               # encoding: [0xb9,0x89,0x00,0x0f]
#CHECK: slbgr	%r15, %r0               # encoding: [0xb9,0x89,0x00,0xf0]
#CHECK: slbgr	%r7, %r8                # encoding: [0xb9,0x89,0x00,0x78]

	slbgr	%r0,%r0
	slbgr	%r0,%r15
	slbgr	%r15,%r0
	slbgr	%r7,%r8
