# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: mdbr	%f0, %f0                # encoding: [0xb3,0x1c,0x00,0x00]
#CHECK: mdbr	%f0, %f15               # encoding: [0xb3,0x1c,0x00,0x0f]
#CHECK: mdbr	%f7, %f8                # encoding: [0xb3,0x1c,0x00,0x78]
#CHECK: mdbr	%f15, %f0               # encoding: [0xb3,0x1c,0x00,0xf0]

	mdbr	%f0, %f0
	mdbr	%f0, %f15
	mdbr	%f7, %f8
	mdbr	%f15, %f0
