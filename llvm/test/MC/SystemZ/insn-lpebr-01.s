# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lpebr	%f0, %f9                # encoding: [0xb3,0x00,0x00,0x09]
#CHECK: lpebr	%f0, %f15               # encoding: [0xb3,0x00,0x00,0x0f]
#CHECK: lpebr	%f15, %f0               # encoding: [0xb3,0x00,0x00,0xf0]
#CHECK: lpebr	%f15, %f9               # encoding: [0xb3,0x00,0x00,0xf9]

	lpebr	%f0,%f9
	lpebr	%f0,%f15
	lpebr	%f15,%f0
	lpebr	%f15,%f9
