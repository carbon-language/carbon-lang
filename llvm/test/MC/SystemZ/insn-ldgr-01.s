# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ldgr	%f0, %r0                # encoding: [0xb3,0xc1,0x00,0x00]
#CHECK: ldgr	%f0, %r15               # encoding: [0xb3,0xc1,0x00,0x0f]
#CHECK: ldgr	%f15, %r0               # encoding: [0xb3,0xc1,0x00,0xf0]
#CHECK: ldgr	%f7, %r9                # encoding: [0xb3,0xc1,0x00,0x79]
#CHECK: ldgr	%f15, %r15              # encoding: [0xb3,0xc1,0x00,0xff]

	ldgr	%f0,%r0
	ldgr	%f0,%r15
	ldgr	%f15,%r0
	ldgr	%f7,%r9
	ldgr	%f15,%r15
