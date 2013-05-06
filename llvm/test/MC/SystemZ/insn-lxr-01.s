# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lxr	%f0, %f8                # encoding: [0xb3,0x65,0x00,0x08]
#CHECK: lxr	%f0, %f13               # encoding: [0xb3,0x65,0x00,0x0d]
#CHECK: lxr	%f13, %f0               # encoding: [0xb3,0x65,0x00,0xd0]
#CHECK: lxr	%f13, %f9               # encoding: [0xb3,0x65,0x00,0xd9]

	lxr	%f0,%f8
	lxr	%f0,%f13
	lxr	%f13,%f0
	lxr	%f13,%f9
