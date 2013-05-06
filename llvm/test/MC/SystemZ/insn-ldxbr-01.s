# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ldxbr	%f0, %f0                # encoding: [0xb3,0x45,0x00,0x00]
#CHECK: ldxbr	%f0, %f13               # encoding: [0xb3,0x45,0x00,0x0d]
#CHECK: ldxbr	%f8, %f12               # encoding: [0xb3,0x45,0x00,0x8c]
#CHECK: ldxbr	%f13, %f0               # encoding: [0xb3,0x45,0x00,0xd0]
#CHECK: ldxbr	%f13, %f13              # encoding: [0xb3,0x45,0x00,0xdd]

	ldxbr	%f0, %f0
	ldxbr	%f0, %f13
	ldxbr	%f8, %f12
	ldxbr	%f13, %f0
	ldxbr	%f13, %f13
