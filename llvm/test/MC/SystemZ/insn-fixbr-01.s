# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: fixbr	%f0, 0, %f0             # encoding: [0xb3,0x47,0x00,0x00]
#CHECK: fixbr	%f0, 0, %f13            # encoding: [0xb3,0x47,0x00,0x0d]
#CHECK: fixbr	%f0, 15, %f0            # encoding: [0xb3,0x47,0xf0,0x00]
#CHECK: fixbr	%f4, 5, %f8             # encoding: [0xb3,0x47,0x50,0x48]
#CHECK: fixbr	%f13, 0, %f0            # encoding: [0xb3,0x47,0x00,0xd0]

	fixbr	%f0, 0, %f0
	fixbr	%f0, 0, %f13
	fixbr	%f0, 15, %f0
	fixbr	%f4, 5, %f8
	fixbr	%f13, 0, %f0
