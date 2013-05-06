# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cgxbr	%r0, 0, %f0             # encoding: [0xb3,0xaa,0x00,0x00]
#CHECK: cgxbr	%r0, 0, %f13            # encoding: [0xb3,0xaa,0x00,0x0d]
#CHECK: cgxbr	%r0, 15, %f0            # encoding: [0xb3,0xaa,0xf0,0x00]
#CHECK: cgxbr	%r4, 5, %f8             # encoding: [0xb3,0xaa,0x50,0x48]
#CHECK: cgxbr	%r15, 0, %f0            # encoding: [0xb3,0xaa,0x00,0xf0]

	cgxbr	%r0, 0, %f0
	cgxbr	%r0, 0, %f13
	cgxbr	%r0, 15, %f0
	cgxbr	%r4, 5, %f8
	cgxbr	%r15, 0, %f0
