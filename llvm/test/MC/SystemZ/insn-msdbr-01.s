# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: msdbr	%f0, %f0, %f0           # encoding: [0xb3,0x1f,0x00,0x00]
#CHECK: msdbr	%f0, %f0, %f15          # encoding: [0xb3,0x1f,0x00,0x0f]
#CHECK: msdbr	%f0, %f15, %f0          # encoding: [0xb3,0x1f,0x00,0xf0]
#CHECK: msdbr	%f15, %f0, %f0          # encoding: [0xb3,0x1f,0xf0,0x00]
#CHECK: msdbr	%f7, %f8, %f9           # encoding: [0xb3,0x1f,0x70,0x89]
#CHECK: msdbr	%f15, %f15, %f15        # encoding: [0xb3,0x1f,0xf0,0xff]

	msdbr	%f0, %f0, %f0
	msdbr	%f0, %f0, %f15
	msdbr	%f0, %f15, %f0
	msdbr	%f15, %f0, %f0
	msdbr	%f7, %f8, %f9
	msdbr	%f15, %f15, %f15
