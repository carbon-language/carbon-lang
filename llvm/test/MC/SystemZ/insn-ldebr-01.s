# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ldebr	%f0, %f15               # encoding: [0xb3,0x04,0x00,0x0f]
#CHECK: ldebr	%f7, %f8                # encoding: [0xb3,0x04,0x00,0x78]
#CHECK: ldebr	%f15, %f0               # encoding: [0xb3,0x04,0x00,0xf0]

	ldebr	%f0, %f15
	ldebr	%f7, %f8
	ldebr	%f15, %f0
