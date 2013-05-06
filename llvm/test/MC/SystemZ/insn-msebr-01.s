# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: msebr	%f0, %f0, %f0           # encoding: [0xb3,0x0f,0x00,0x00]
#CHECK: msebr	%f0, %f0, %f15          # encoding: [0xb3,0x0f,0x00,0x0f]
#CHECK: msebr	%f0, %f15, %f0          # encoding: [0xb3,0x0f,0x00,0xf0]
#CHECK: msebr	%f15, %f0, %f0          # encoding: [0xb3,0x0f,0xf0,0x00]
#CHECK: msebr	%f7, %f8, %f9           # encoding: [0xb3,0x0f,0x70,0x89]
#CHECK: msebr	%f15, %f15, %f15        # encoding: [0xb3,0x0f,0xf0,0xff]

	msebr	%f0, %f0, %f0
	msebr	%f0, %f0, %f15
	msebr	%f0, %f15, %f0
	msebr	%f15, %f0, %f0
	msebr	%f7, %f8, %f9
	msebr	%f15, %f15, %f15
