# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: br	%r1                     # encoding: [0x07,0xf1]
#CHECK: br	%r14                    # encoding: [0x07,0xfe]
#CHECK: br	%r15                    # encoding: [0x07,0xff]

	br	%r1
	br	%r14
	br	%r15
