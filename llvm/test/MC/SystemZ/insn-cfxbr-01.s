# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cfxbr	%r0, 0, %f0             # encoding: [0xb3,0x9a,0x00,0x00]
#CHECK: cfxbr	%r0, 0, %f13            # encoding: [0xb3,0x9a,0x00,0x0d]
#CHECK: cfxbr	%r0, 15, %f0            # encoding: [0xb3,0x9a,0xf0,0x00]
#CHECK: cfxbr	%r4, 5, %f8             # encoding: [0xb3,0x9a,0x50,0x48]
#CHECK: cfxbr	%r15, 0, %f0            # encoding: [0xb3,0x9a,0x00,0xf0]

	cfxbr	%r0, 0, %f0
	cfxbr	%r0, 0, %f13
	cfxbr	%r0, 15, %f0
	cfxbr	%r4, 5, %f8
	cfxbr	%r15, 0, %f0
