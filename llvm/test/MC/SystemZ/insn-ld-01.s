# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ld	%f0, 0                  # encoding: [0x68,0x00,0x00,0x00]
#CHECK: ld	%f0, 4095               # encoding: [0x68,0x00,0x0f,0xff]
#CHECK: ld	%f0, 0(%r1)             # encoding: [0x68,0x00,0x10,0x00]
#CHECK: ld	%f0, 0(%r15)            # encoding: [0x68,0x00,0xf0,0x00]
#CHECK: ld	%f0, 4095(%r1,%r15)     # encoding: [0x68,0x01,0xff,0xff]
#CHECK: ld	%f0, 4095(%r15,%r1)     # encoding: [0x68,0x0f,0x1f,0xff]
#CHECK: ld	%f15, 0                 # encoding: [0x68,0xf0,0x00,0x00]

	ld	%f0, 0
	ld	%f0, 4095
	ld	%f0, 0(%r1)
	ld	%f0, 0(%r15)
	ld	%f0, 4095(%r1,%r15)
	ld	%f0, 4095(%r15,%r1)
	ld	%f15, 0
