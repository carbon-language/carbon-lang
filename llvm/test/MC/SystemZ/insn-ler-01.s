# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ler	%f0, %f9                # encoding: [0x38,0x09]
#CHECK: ler	%f0, %f15               # encoding: [0x38,0x0f]
#CHECK: ler	%f15, %f0               # encoding: [0x38,0xf0]
#CHECK: ler	%f15, %f9               # encoding: [0x38,0xf9]

	ler	%f0,%f9
	ler	%f0,%f15
	ler	%f15,%f0
	ler	%f15,%f9
