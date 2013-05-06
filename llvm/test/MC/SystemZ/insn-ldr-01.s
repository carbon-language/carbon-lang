# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ldr	%f0, %f9                # encoding: [0x28,0x09]
#CHECK: ldr	%f0, %f15               # encoding: [0x28,0x0f]
#CHECK: ldr	%f15, %f0               # encoding: [0x28,0xf0]
#CHECK: ldr	%f15, %f9               # encoding: [0x28,0xf9]

	ldr	%f0,%f9
	ldr	%f0,%f15
	ldr	%f15,%f0
	ldr	%f15,%f9
