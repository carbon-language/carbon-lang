# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lcebr	%f0, %f9                # encoding: [0xb3,0x03,0x00,0x09]
#CHECK: lcebr	%f0, %f15               # encoding: [0xb3,0x03,0x00,0x0f]
#CHECK: lcebr	%f15, %f0               # encoding: [0xb3,0x03,0x00,0xf0]
#CHECK: lcebr	%f15, %f9               # encoding: [0xb3,0x03,0x00,0xf9]

	lcebr	%f0,%f9
	lcebr	%f0,%f15
	lcebr	%f15,%f0
	lcebr	%f15,%f9
