# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lzxr	%f0                     # encoding: [0xb3,0x76,0x00,0x00]
#CHECK: lzxr	%f8                     # encoding: [0xb3,0x76,0x00,0x80]
#CHECK: lzxr	%f13                    # encoding: [0xb3,0x76,0x00,0xd0]

	lzxr	%f0
	lzxr	%f8
	lzxr	%f13
