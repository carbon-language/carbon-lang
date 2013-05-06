# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cfebr	%r0, 0, %f0             # encoding: [0xb3,0x98,0x00,0x00]
#CHECK: cfebr	%r0, 0, %f15            # encoding: [0xb3,0x98,0x00,0x0f]
#CHECK: cfebr	%r0, 15, %f0            # encoding: [0xb3,0x98,0xf0,0x00]
#CHECK: cfebr	%r4, 5, %f6             # encoding: [0xb3,0x98,0x50,0x46]
#CHECK: cfebr	%r15, 0, %f0            # encoding: [0xb3,0x98,0x00,0xf0]

	cfebr	%r0, 0, %f0
	cfebr	%r0, 0, %f15
	cfebr	%r0, 15, %f0
	cfebr	%r4, 5, %f6
	cfebr	%r15, 0, %f0
