# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lexbr	%f0, %f0                # encoding: [0xb3,0x46,0x00,0x00]
#CHECK: lexbr	%f0, %f13               # encoding: [0xb3,0x46,0x00,0x0d]
#CHECK: lexbr	%f8, %f12               # encoding: [0xb3,0x46,0x00,0x8c]
#CHECK: lexbr	%f13, %f0               # encoding: [0xb3,0x46,0x00,0xd0]
#CHECK: lexbr	%f13, %f13              # encoding: [0xb3,0x46,0x00,0xdd]

	lexbr	%f0, %f0
	lexbr	%f0, %f13
	lexbr	%f8, %f12
	lexbr	%f13, %f0
	lexbr	%f13, %f13
