# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lzdr	%f0                     # encoding: [0xb3,0x75,0x00,0x00]
#CHECK: lzdr	%f7                     # encoding: [0xb3,0x75,0x00,0x70]
#CHECK: lzdr	%f15                    # encoding: [0xb3,0x75,0x00,0xf0]

	lzdr	%f0
	lzdr	%f7
	lzdr	%f15
