# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: xgr	%r0, %r0                # encoding: [0xb9,0x82,0x00,0x00]
#CHECK: xgr	%r0, %r15               # encoding: [0xb9,0x82,0x00,0x0f]
#CHECK: xgr	%r15, %r0               # encoding: [0xb9,0x82,0x00,0xf0]
#CHECK: xgr	%r7, %r8                # encoding: [0xb9,0x82,0x00,0x78]

	xgr	%r0,%r0
	xgr	%r0,%r15
	xgr	%r15,%r0
	xgr	%r7,%r8
