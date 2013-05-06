# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lcdbr	%f0, %f9                # encoding: [0xb3,0x13,0x00,0x09]
#CHECK: lcdbr	%f0, %f15               # encoding: [0xb3,0x13,0x00,0x0f]
#CHECK: lcdbr	%f15, %f0               # encoding: [0xb3,0x13,0x00,0xf0]
#CHECK: lcdbr	%f15, %f9               # encoding: [0xb3,0x13,0x00,0xf9]

	lcdbr	%f0,%f9
	lcdbr	%f0,%f15
	lcdbr	%f15,%f0
	lcdbr	%f15,%f9
