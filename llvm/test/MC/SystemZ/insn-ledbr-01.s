# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ledbr	%f0, %f0                # encoding: [0xb3,0x44,0x00,0x00]
#CHECK: ledbr	%f0, %f15               # encoding: [0xb3,0x44,0x00,0x0f]
#CHECK: ledbr	%f7, %f8                # encoding: [0xb3,0x44,0x00,0x78]
#CHECK: ledbr	%f15, %f0               # encoding: [0xb3,0x44,0x00,0xf0]
#CHECK: ledbr	%f15, %f15              # encoding: [0xb3,0x44,0x00,0xff]

	ledbr	%f0, %f0
	ledbr	%f0, %f15
	ledbr	%f7, %f8
	ledbr	%f15, %f0
	ledbr	%f15, %f15
