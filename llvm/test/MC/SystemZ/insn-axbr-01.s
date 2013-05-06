# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: axbr	%f0, %f0                # encoding: [0xb3,0x4a,0x00,0x00]
#CHECK: axbr	%f0, %f13               # encoding: [0xb3,0x4a,0x00,0x0d]
#CHECK: axbr	%f8, %f8                # encoding: [0xb3,0x4a,0x00,0x88]
#CHECK: axbr	%f13, %f0               # encoding: [0xb3,0x4a,0x00,0xd0]

	axbr	%f0, %f0
	axbr	%f0, %f13
	axbr	%f8, %f8
	axbr	%f13, %f0
