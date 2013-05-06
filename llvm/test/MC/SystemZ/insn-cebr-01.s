# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cebr	%f0, %f0                # encoding: [0xb3,0x09,0x00,0x00]
#CHECK: cebr	%f0, %f15               # encoding: [0xb3,0x09,0x00,0x0f]
#CHECK: cebr	%f7, %f8                # encoding: [0xb3,0x09,0x00,0x78]
#CHECK: cebr	%f15, %f0               # encoding: [0xb3,0x09,0x00,0xf0]

	cebr	%f0, %f0
	cebr	%f0, %f15
	cebr	%f7, %f8
	cebr	%f15, %f0
