# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cxgbr	%f0, %r0                # encoding: [0xb3,0xa6,0x00,0x00]
#CHECK: cxgbr	%f0, %r15               # encoding: [0xb3,0xa6,0x00,0x0f]
#CHECK: cxgbr	%f13, %r0               # encoding: [0xb3,0xa6,0x00,0xd0]
#CHECK: cxgbr	%f8, %r7                # encoding: [0xb3,0xa6,0x00,0x87]
#CHECK: cxgbr	%f13, %r15              # encoding: [0xb3,0xa6,0x00,0xdf]

	cxgbr	%f0, %r0
	cxgbr	%f0, %r15
	cxgbr	%f13, %r0
	cxgbr	%f8, %r7
	cxgbr	%f13, %r15
