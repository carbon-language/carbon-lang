# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: meebr	%f0, %f0                # encoding: [0xb3,0x17,0x00,0x00]
#CHECK: meebr	%f0, %f15               # encoding: [0xb3,0x17,0x00,0x0f]
#CHECK: meebr	%f7, %f8                # encoding: [0xb3,0x17,0x00,0x78]
#CHECK: meebr	%f15, %f0               # encoding: [0xb3,0x17,0x00,0xf0]

	meebr	%f0, %f0
	meebr	%f0, %f15
	meebr	%f7, %f8
	meebr	%f15, %f0
