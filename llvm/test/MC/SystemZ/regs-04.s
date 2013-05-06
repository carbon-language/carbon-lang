# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ler	%f0, %f1                # encoding: [0x38,0x01]
#CHECK: ler	%f2, %f3                # encoding: [0x38,0x23]
#CHECK: ler	%f4, %f5                # encoding: [0x38,0x45]
#CHECK: ler	%f6, %f7                # encoding: [0x38,0x67]
#CHECK: ler	%f8, %f9                # encoding: [0x38,0x89]
#CHECK: ler	%f10, %f11              # encoding: [0x38,0xab]
#CHECK: ler	%f12, %f13              # encoding: [0x38,0xcd]
#CHECK: ler	%f14, %f15              # encoding: [0x38,0xef]

	ler	%f0,%f1
	ler	%f2,%f3
	ler	%f4,%f5
	ler	%f6,%f7
	ler	%f8,%f9
	ler	%f10,%f11
	ler	%f12,%f13
	ler	%f14,%f15
