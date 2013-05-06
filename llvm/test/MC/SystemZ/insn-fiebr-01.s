# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: fiebr	%f0, 0, %f0             # encoding: [0xb3,0x57,0x00,0x00]
#CHECK: fiebr	%f0, 0, %f15            # encoding: [0xb3,0x57,0x00,0x0f]
#CHECK: fiebr	%f0, 15, %f0            # encoding: [0xb3,0x57,0xf0,0x00]
#CHECK: fiebr	%f4, 5, %f6             # encoding: [0xb3,0x57,0x50,0x46]
#CHECK: fiebr	%f15, 0, %f0            # encoding: [0xb3,0x57,0x00,0xf0]

	fiebr	%f0, 0, %f0
	fiebr	%f0, 0, %f15
	fiebr	%f0, 15, %f0
	fiebr	%f4, 5, %f6
	fiebr	%f15, 0, %f0
