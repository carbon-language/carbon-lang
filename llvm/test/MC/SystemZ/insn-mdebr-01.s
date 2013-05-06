# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: mdebr	%f0, %f0                # encoding: [0xb3,0x0c,0x00,0x00]
#CHECK: mdebr	%f0, %f15               # encoding: [0xb3,0x0c,0x00,0x0f]
#CHECK: mdebr	%f7, %f8                # encoding: [0xb3,0x0c,0x00,0x78]
#CHECK: mdebr	%f15, %f0               # encoding: [0xb3,0x0c,0x00,0xf0]

	mdebr	%f0, %f0
	mdebr	%f0, %f15
	mdebr	%f7, %f8
	mdebr	%f15, %f0
