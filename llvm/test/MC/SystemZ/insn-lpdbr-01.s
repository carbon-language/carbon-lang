# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lpdbr	%f0, %f9                # encoding: [0xb3,0x10,0x00,0x09]
#CHECK: lpdbr	%f0, %f15               # encoding: [0xb3,0x10,0x00,0x0f]
#CHECK: lpdbr	%f15, %f0               # encoding: [0xb3,0x10,0x00,0xf0]
#CHECK: lpdbr	%f15, %f9               # encoding: [0xb3,0x10,0x00,0xf9]

	lpdbr	%f0,%f9
	lpdbr	%f0,%f15
	lpdbr	%f15,%f0
	lpdbr	%f15,%f9
