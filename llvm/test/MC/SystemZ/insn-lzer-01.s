# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lzer	%f0                     # encoding: [0xb3,0x74,0x00,0x00]
#CHECK: lzer	%f7                     # encoding: [0xb3,0x74,0x00,0x70]
#CHECK: lzer	%f15                    # encoding: [0xb3,0x74,0x00,0xf0]

	lzer	%f0
	lzer	%f7
	lzer	%f15
