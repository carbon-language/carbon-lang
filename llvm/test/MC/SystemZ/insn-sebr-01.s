# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sebr	%f0, %f0                # encoding: [0xb3,0x0b,0x00,0x00]
#CHECK: sebr	%f0, %f15               # encoding: [0xb3,0x0b,0x00,0x0f]
#CHECK: sebr	%f7, %f8                # encoding: [0xb3,0x0b,0x00,0x78]
#CHECK: sebr	%f15, %f0               # encoding: [0xb3,0x0b,0x00,0xf0]

	sebr	%f0, %f0
	sebr	%f0, %f15
	sebr	%f7, %f8
	sebr	%f15, %f0
