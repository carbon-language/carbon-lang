# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ch	%r0, 0                  # encoding: [0x49,0x00,0x00,0x00]
#CHECK: ch	%r0, 4095               # encoding: [0x49,0x00,0x0f,0xff]
#CHECK: ch	%r0, 0(%r1)             # encoding: [0x49,0x00,0x10,0x00]
#CHECK: ch	%r0, 0(%r15)            # encoding: [0x49,0x00,0xf0,0x00]
#CHECK: ch	%r0, 4095(%r1,%r15)     # encoding: [0x49,0x01,0xff,0xff]
#CHECK: ch	%r0, 4095(%r15,%r1)     # encoding: [0x49,0x0f,0x1f,0xff]
#CHECK: ch	%r15, 0                 # encoding: [0x49,0xf0,0x00,0x00]

	ch	%r0, 0
	ch	%r0, 4095
	ch	%r0, 0(%r1)
	ch	%r0, 0(%r15)
	ch	%r0, 4095(%r1,%r15)
	ch	%r0, 4095(%r15,%r1)
	ch	%r15, 0
