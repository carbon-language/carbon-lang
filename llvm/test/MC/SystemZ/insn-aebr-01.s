# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: aebr	%f0, %f0                # encoding: [0xb3,0x0a,0x00,0x00]
#CHECK: aebr	%f0, %f15               # encoding: [0xb3,0x0a,0x00,0x0f]
#CHECK: aebr	%f7, %f8                # encoding: [0xb3,0x0a,0x00,0x78]
#CHECK: aebr	%f15, %f0               # encoding: [0xb3,0x0a,0x00,0xf0]

	aebr	%f0, %f0
	aebr	%f0, %f15
	aebr	%f7, %f8
	aebr	%f15, %f0
