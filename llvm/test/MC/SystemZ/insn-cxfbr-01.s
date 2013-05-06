# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cxfbr	%f0, %r0                # encoding: [0xb3,0x96,0x00,0x00]
#CHECK: cxfbr	%f0, %r15               # encoding: [0xb3,0x96,0x00,0x0f]
#CHECK: cxfbr	%f13, %r0               # encoding: [0xb3,0x96,0x00,0xd0]
#CHECK: cxfbr	%f8, %r7                # encoding: [0xb3,0x96,0x00,0x87]
#CHECK: cxfbr	%f13, %r15              # encoding: [0xb3,0x96,0x00,0xdf]

	cxfbr	%f0, %r0
	cxfbr	%f0, %r15
	cxfbr	%f13, %r0
	cxfbr	%f8, %r7
	cxfbr	%f13, %r15
