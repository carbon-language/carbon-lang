# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: le	%f0, 0                  # encoding: [0x78,0x00,0x00,0x00]
#CHECK: le	%f0, 4095               # encoding: [0x78,0x00,0x0f,0xff]
#CHECK: le	%f0, 0(%r1)             # encoding: [0x78,0x00,0x10,0x00]
#CHECK: le	%f0, 0(%r15)            # encoding: [0x78,0x00,0xf0,0x00]
#CHECK: le	%f0, 4095(%r1,%r15)     # encoding: [0x78,0x01,0xff,0xff]
#CHECK: le	%f0, 4095(%r15,%r1)     # encoding: [0x78,0x0f,0x1f,0xff]
#CHECK: le	%f15, 0                 # encoding: [0x78,0xf0,0x00,0x00]

	le	%f0, 0
	le	%f0, 4095
	le	%f0, 0(%r1)
	le	%f0, 0(%r15)
	le	%f0, 4095(%r1,%r15)
	le	%f0, 4095(%r15,%r1)
	le	%f15, 0
