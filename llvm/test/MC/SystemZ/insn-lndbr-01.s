# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lndbr	%f0, %f9                # encoding: [0xb3,0x11,0x00,0x09]
#CHECK: lndbr	%f0, %f15               # encoding: [0xb3,0x11,0x00,0x0f]
#CHECK: lndbr	%f15, %f0               # encoding: [0xb3,0x11,0x00,0xf0]
#CHECK: lndbr	%f15, %f9               # encoding: [0xb3,0x11,0x00,0xf9]

	lndbr	%f0,%f9
	lndbr	%f0,%f15
	lndbr	%f15,%f0
	lndbr	%f15,%f9
