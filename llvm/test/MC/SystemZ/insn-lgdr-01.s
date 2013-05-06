# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lgdr	%r0, %f0                # encoding: [0xb3,0xcd,0x00,0x00]
#CHECK: lgdr	%r0, %f15               # encoding: [0xb3,0xcd,0x00,0x0f]
#CHECK: lgdr	%r15, %f0               # encoding: [0xb3,0xcd,0x00,0xf0]
#CHECK: lgdr	%r8, %f8                # encoding: [0xb3,0xcd,0x00,0x88]
#CHECK: lgdr	%r15, %f15              # encoding: [0xb3,0xcd,0x00,0xff]

	lgdr	%r0,%f0
	lgdr	%r0,%f15
	lgdr	%r15,%f0
	lgdr	%r8,%f8
	lgdr	%r15,%f15
