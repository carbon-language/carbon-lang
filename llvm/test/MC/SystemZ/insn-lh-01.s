# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lh	%r0, 0                  # encoding: [0x48,0x00,0x00,0x00]
#CHECK: lh	%r0, 4095               # encoding: [0x48,0x00,0x0f,0xff]
#CHECK: lh	%r0, 0(%r1)             # encoding: [0x48,0x00,0x10,0x00]
#CHECK: lh	%r0, 0(%r15)            # encoding: [0x48,0x00,0xf0,0x00]
#CHECK: lh	%r0, 4095(%r1,%r15)     # encoding: [0x48,0x01,0xff,0xff]
#CHECK: lh	%r0, 4095(%r15,%r1)     # encoding: [0x48,0x0f,0x1f,0xff]
#CHECK: lh	%r15, 0                 # encoding: [0x48,0xf0,0x00,0x00]

	lh	%r0, 0
	lh	%r0, 4095
	lh	%r0, 0(%r1)
	lh	%r0, 0(%r15)
	lh	%r0, 4095(%r1,%r15)
	lh	%r0, 4095(%r15,%r1)
	lh	%r15, 0
