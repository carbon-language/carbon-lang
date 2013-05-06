# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lpxbr	%f0, %f8                # encoding: [0xb3,0x40,0x00,0x08]
#CHECK: lpxbr	%f0, %f13               # encoding: [0xb3,0x40,0x00,0x0d]
#CHECK: lpxbr	%f13, %f0               # encoding: [0xb3,0x40,0x00,0xd0]
#CHECK: lpxbr	%f13, %f9               # encoding: [0xb3,0x40,0x00,0xd9]

	lpxbr	%f0,%f8
	lpxbr	%f0,%f13
	lpxbr	%f13,%f0
	lpxbr	%f13,%f9
