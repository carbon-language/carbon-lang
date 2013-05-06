# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cegbr	%f0, %r0                # encoding: [0xb3,0xa4,0x00,0x00]
#CHECK: cegbr	%f0, %r15               # encoding: [0xb3,0xa4,0x00,0x0f]
#CHECK: cegbr	%f15, %r0               # encoding: [0xb3,0xa4,0x00,0xf0]
#CHECK: cegbr	%f7, %r8                # encoding: [0xb3,0xa4,0x00,0x78]
#CHECK: cegbr	%f15, %r15              # encoding: [0xb3,0xa4,0x00,0xff]

	cegbr	%f0, %r0
	cegbr	%f0, %r15
	cegbr	%f15, %r0
	cegbr	%f7, %r8
	cegbr	%f15, %r15
