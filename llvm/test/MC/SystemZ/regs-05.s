# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ldr	%f0, %f1                # encoding: [0x28,0x01]
#CHECK: ldr	%f2, %f3                # encoding: [0x28,0x23]
#CHECK: ldr	%f4, %f5                # encoding: [0x28,0x45]
#CHECK: ldr	%f6, %f7                # encoding: [0x28,0x67]
#CHECK: ldr	%f8, %f9                # encoding: [0x28,0x89]
#CHECK: ldr	%f10, %f11              # encoding: [0x28,0xab]
#CHECK: ldr	%f12, %f13              # encoding: [0x28,0xcd]
#CHECK: ldr	%f14, %f15              # encoding: [0x28,0xef]

	ldr	%f0,%f1
	ldr	%f2,%f3
	ldr	%f4,%f5
	ldr	%f6,%f7
	ldr	%f8,%f9
	ldr	%f10,%f11
	ldr	%f12,%f13
	ldr	%f14,%f15
