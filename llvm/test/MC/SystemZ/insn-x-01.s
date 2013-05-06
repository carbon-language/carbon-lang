# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: x	%r0, 0                  # encoding: [0x57,0x00,0x00,0x00]
#CHECK: x	%r0, 4095               # encoding: [0x57,0x00,0x0f,0xff]
#CHECK: x	%r0, 0(%r1)             # encoding: [0x57,0x00,0x10,0x00]
#CHECK: x	%r0, 0(%r15)            # encoding: [0x57,0x00,0xf0,0x00]
#CHECK: x	%r0, 4095(%r1,%r15)     # encoding: [0x57,0x01,0xff,0xff]
#CHECK: x	%r0, 4095(%r15,%r1)     # encoding: [0x57,0x0f,0x1f,0xff]
#CHECK: x	%r15, 0                 # encoding: [0x57,0xf0,0x00,0x00]

	x	%r0, 0
	x	%r0, 4095
	x	%r0, 0(%r1)
	x	%r0, 0(%r15)
	x	%r0, 4095(%r1,%r15)
	x	%r0, 4095(%r15,%r1)
	x	%r15, 0
