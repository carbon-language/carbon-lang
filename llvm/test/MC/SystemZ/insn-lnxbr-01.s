# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lnxbr	%f0, %f8                # encoding: [0xb3,0x41,0x00,0x08]
#CHECK: lnxbr	%f0, %f13               # encoding: [0xb3,0x41,0x00,0x0d]
#CHECK: lnxbr	%f13, %f0               # encoding: [0xb3,0x41,0x00,0xd0]
#CHECK: lnxbr	%f13, %f9               # encoding: [0xb3,0x41,0x00,0xd9]

	lnxbr	%f0,%f8
	lnxbr	%f0,%f13
	lnxbr	%f13,%f0
	lnxbr	%f13,%f9
