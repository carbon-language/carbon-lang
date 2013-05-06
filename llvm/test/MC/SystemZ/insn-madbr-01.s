# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: madbr	%f0, %f0, %f0           # encoding: [0xb3,0x1e,0x00,0x00]
#CHECK: madbr	%f0, %f0, %f15          # encoding: [0xb3,0x1e,0x00,0x0f]
#CHECK: madbr	%f0, %f15, %f0          # encoding: [0xb3,0x1e,0x00,0xf0]
#CHECK: madbr	%f15, %f0, %f0          # encoding: [0xb3,0x1e,0xf0,0x00]
#CHECK: madbr	%f7, %f8, %f9           # encoding: [0xb3,0x1e,0x70,0x89]
#CHECK: madbr	%f15, %f15, %f15        # encoding: [0xb3,0x1e,0xf0,0xff]

	madbr	%f0, %f0, %f0
	madbr	%f0, %f0, %f15
	madbr	%f0, %f15, %f0
	madbr	%f15, %f0, %f0
	madbr	%f7, %f8, %f9
	madbr	%f15, %f15, %f15
