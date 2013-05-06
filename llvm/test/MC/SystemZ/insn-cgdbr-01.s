# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cgdbr	%r0, 0, %f0             # encoding: [0xb3,0xa9,0x00,0x00]
#CHECK: cgdbr	%r0, 0, %f15            # encoding: [0xb3,0xa9,0x00,0x0f]
#CHECK: cgdbr	%r0, 15, %f0            # encoding: [0xb3,0xa9,0xf0,0x00]
#CHECK: cgdbr	%r4, 5, %f6             # encoding: [0xb3,0xa9,0x50,0x46]
#CHECK: cgdbr	%r15, 0, %f0            # encoding: [0xb3,0xa9,0x00,0xf0]

	cgdbr	%r0, 0, %f0
	cgdbr	%r0, 0, %f15
	cgdbr	%r0, 15, %f0
	cgdbr	%r4, 5, %f6
	cgdbr	%r15, 0, %f0
