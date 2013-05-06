# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lxr	%f0, %f1                # encoding: [0xb3,0x65,0x00,0x01]
#CHECK: lxr	%f4, %f5                # encoding: [0xb3,0x65,0x00,0x45]
#CHECK: lxr	%f8, %f9                # encoding: [0xb3,0x65,0x00,0x89]
#CHECK: lxr	%f12, %f13              # encoding: [0xb3,0x65,0x00,0xcd]

	lxr	%f0,%f1
	lxr	%f4,%f5
	lxr	%f8,%f9
	lxr	%f12,%f13
