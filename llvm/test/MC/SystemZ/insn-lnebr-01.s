# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lnebr	%f0, %f9                # encoding: [0xb3,0x01,0x00,0x09]
#CHECK: lnebr	%f0, %f15               # encoding: [0xb3,0x01,0x00,0x0f]
#CHECK: lnebr	%f15, %f0               # encoding: [0xb3,0x01,0x00,0xf0]
#CHECK: lnebr	%f15, %f9               # encoding: [0xb3,0x01,0x00,0xf9]

	lnebr	%f0,%f9
	lnebr	%f0,%f15
	lnebr	%f15,%f0
	lnebr	%f15,%f9
