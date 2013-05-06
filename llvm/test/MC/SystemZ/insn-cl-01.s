# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cl	%r0, 0                  # encoding: [0x55,0x00,0x00,0x00]
#CHECK: cl	%r0, 4095               # encoding: [0x55,0x00,0x0f,0xff]
#CHECK: cl	%r0, 0(%r1)             # encoding: [0x55,0x00,0x10,0x00]
#CHECK: cl	%r0, 0(%r15)            # encoding: [0x55,0x00,0xf0,0x00]
#CHECK: cl	%r0, 4095(%r1,%r15)     # encoding: [0x55,0x01,0xff,0xff]
#CHECK: cl	%r0, 4095(%r15,%r1)     # encoding: [0x55,0x0f,0x1f,0xff]
#CHECK: cl	%r15, 0                 # encoding: [0x55,0xf0,0x00,0x00]

	cl	%r0, 0
	cl	%r0, 4095
	cl	%r0, 0(%r1)
	cl	%r0, 0(%r15)
	cl	%r0, 4095(%r1,%r15)
	cl	%r0, 4095(%r15,%r1)
	cl	%r15, 0
