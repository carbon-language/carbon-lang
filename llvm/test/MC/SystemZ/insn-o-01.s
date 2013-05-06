# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: o	%r0, 0                  # encoding: [0x56,0x00,0x00,0x00]
#CHECK: o	%r0, 4095               # encoding: [0x56,0x00,0x0f,0xff]
#CHECK: o	%r0, 0(%r1)             # encoding: [0x56,0x00,0x10,0x00]
#CHECK: o	%r0, 0(%r15)            # encoding: [0x56,0x00,0xf0,0x00]
#CHECK: o	%r0, 4095(%r1,%r15)     # encoding: [0x56,0x01,0xff,0xff]
#CHECK: o	%r0, 4095(%r15,%r1)     # encoding: [0x56,0x0f,0x1f,0xff]
#CHECK: o	%r15, 0                 # encoding: [0x56,0xf0,0x00,0x00]

	o	%r0, 0
	o	%r0, 4095
	o	%r0, 0(%r1)
	o	%r0, 0(%r15)
	o	%r0, 4095(%r1,%r15)
	o	%r0, 4095(%r15,%r1)
	o	%r15, 0
