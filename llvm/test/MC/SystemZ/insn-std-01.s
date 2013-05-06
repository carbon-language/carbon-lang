# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: std	%f0, 0                  # encoding: [0x60,0x00,0x00,0x00]
#CHECK: std	%f0, 4095               # encoding: [0x60,0x00,0x0f,0xff]
#CHECK: std	%f0, 0(%r1)             # encoding: [0x60,0x00,0x10,0x00]
#CHECK: std	%f0, 0(%r15)            # encoding: [0x60,0x00,0xf0,0x00]
#CHECK: std	%f0, 4095(%r1,%r15)     # encoding: [0x60,0x01,0xff,0xff]
#CHECK: std	%f0, 4095(%r15,%r1)     # encoding: [0x60,0x0f,0x1f,0xff]
#CHECK: std	%f15, 0                 # encoding: [0x60,0xf0,0x00,0x00]

	std	%f0, 0
	std	%f0, 4095
	std	%f0, 0(%r1)
	std	%f0, 0(%r15)
	std	%f0, 4095(%r1,%r15)
	std	%f0, 4095(%r15,%r1)
	std	%f15, 0
