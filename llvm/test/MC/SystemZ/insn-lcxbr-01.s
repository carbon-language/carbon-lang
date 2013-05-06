# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lcxbr	%f0, %f8                # encoding: [0xb3,0x43,0x00,0x08]
#CHECK: lcxbr	%f0, %f13               # encoding: [0xb3,0x43,0x00,0x0d]
#CHECK: lcxbr	%f13, %f0               # encoding: [0xb3,0x43,0x00,0xd0]
#CHECK: lcxbr	%f13, %f9               # encoding: [0xb3,0x43,0x00,0xd9]

	lcxbr	%f0,%f8
	lcxbr	%f0,%f13
	lcxbr	%f13,%f0
	lcxbr	%f13,%f9
