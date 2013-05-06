# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: fidbr	%f0, 0, %f0             # encoding: [0xb3,0x5f,0x00,0x00]
#CHECK: fidbr	%f0, 0, %f15            # encoding: [0xb3,0x5f,0x00,0x0f]
#CHECK: fidbr	%f0, 15, %f0            # encoding: [0xb3,0x5f,0xf0,0x00]
#CHECK: fidbr	%f4, 5, %f6             # encoding: [0xb3,0x5f,0x50,0x46]
#CHECK: fidbr	%f15, 0, %f0            # encoding: [0xb3,0x5f,0x00,0xf0]

	fidbr	%f0, 0, %f0
	fidbr	%f0, 0, %f15
	fidbr	%f0, 15, %f0
	fidbr	%f4, 5, %f6
	fidbr	%f15, 0, %f0
