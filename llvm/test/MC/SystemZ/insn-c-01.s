# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: c	%r0, 0                  # encoding: [0x59,0x00,0x00,0x00]
#CHECK: c	%r0, 4095               # encoding: [0x59,0x00,0x0f,0xff]
#CHECK: c	%r0, 0(%r1)             # encoding: [0x59,0x00,0x10,0x00]
#CHECK: c	%r0, 0(%r15)            # encoding: [0x59,0x00,0xf0,0x00]
#CHECK: c	%r0, 4095(%r1,%r15)     # encoding: [0x59,0x01,0xff,0xff]
#CHECK: c	%r0, 4095(%r15,%r1)     # encoding: [0x59,0x0f,0x1f,0xff]
#CHECK: c	%r15, 0                 # encoding: [0x59,0xf0,0x00,0x00]

	c	%r0, 0
	c	%r0, 4095
	c	%r0, 0(%r1)
	c	%r0, 0(%r15)
	c	%r0, 4095(%r1,%r15)
	c	%r0, 4095(%r15,%r1)
	c	%r15, 0
