# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: maebr	%f0, %f0, %f0           # encoding: [0xb3,0x0e,0x00,0x00]
#CHECK: maebr	%f0, %f0, %f15          # encoding: [0xb3,0x0e,0x00,0x0f]
#CHECK: maebr	%f0, %f15, %f0          # encoding: [0xb3,0x0e,0x00,0xf0]
#CHECK: maebr	%f15, %f0, %f0          # encoding: [0xb3,0x0e,0xf0,0x00]
#CHECK: maebr	%f7, %f8, %f9           # encoding: [0xb3,0x0e,0x70,0x89]
#CHECK: maebr	%f15, %f15, %f15        # encoding: [0xb3,0x0e,0xf0,0xff]

	maebr	%f0, %f0, %f0
	maebr	%f0, %f0, %f15
	maebr	%f0, %f15, %f0
	maebr	%f15, %f0, %f0
	maebr	%f7, %f8, %f9
	maebr	%f15, %f15, %f15
