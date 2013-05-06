# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cpsdr	%f0, %f0, %f0           # encoding: [0xb3,0x72,0x00,0x00]
#CHECK: cpsdr	%f0, %f0, %f15          # encoding: [0xb3,0x72,0x00,0x0f]
#CHECK: cpsdr	%f0, %f15, %f0          # encoding: [0xb3,0x72,0xf0,0x00]
#CHECK: cpsdr	%f15, %f0, %f0          # encoding: [0xb3,0x72,0x00,0xf0]
#CHECK: cpsdr	%f1, %f2, %f3           # encoding: [0xb3,0x72,0x20,0x13]
#CHECK: cpsdr	%f15, %f15, %f15        # encoding: [0xb3,0x72,0xf0,0xff]

	cpsdr	%f0, %f0, %f0
	cpsdr	%f0, %f0, %f15
	cpsdr	%f0, %f15, %f0
	cpsdr	%f15, %f0, %f0
	cpsdr	%f1, %f2, %f3
	cpsdr	%f15, %f15, %f15

