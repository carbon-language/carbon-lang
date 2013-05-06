# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: mh	%r0, 0                  # encoding: [0x4c,0x00,0x00,0x00]
#CHECK: mh	%r0, 4095               # encoding: [0x4c,0x00,0x0f,0xff]
#CHECK: mh	%r0, 0(%r1)             # encoding: [0x4c,0x00,0x10,0x00]
#CHECK: mh	%r0, 0(%r15)            # encoding: [0x4c,0x00,0xf0,0x00]
#CHECK: mh	%r0, 4095(%r1,%r15)     # encoding: [0x4c,0x01,0xff,0xff]
#CHECK: mh	%r0, 4095(%r15,%r1)     # encoding: [0x4c,0x0f,0x1f,0xff]
#CHECK: mh	%r15, 0                 # encoding: [0x4c,0xf0,0x00,0x00]

	mh	%r0, 0
	mh	%r0, 4095
	mh	%r0, 0(%r1)
	mh	%r0, 0(%r15)
	mh	%r0, 4095(%r1,%r15)
	mh	%r0, 4095(%r15,%r1)
	mh	%r15, 0
