# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ah	%r0, 0                  # encoding: [0x4a,0x00,0x00,0x00]
#CHECK: ah	%r0, 4095               # encoding: [0x4a,0x00,0x0f,0xff]
#CHECK: ah	%r0, 0(%r1)             # encoding: [0x4a,0x00,0x10,0x00]
#CHECK: ah	%r0, 0(%r15)            # encoding: [0x4a,0x00,0xf0,0x00]
#CHECK: ah	%r0, 4095(%r1,%r15)     # encoding: [0x4a,0x01,0xff,0xff]
#CHECK: ah	%r0, 4095(%r15,%r1)     # encoding: [0x4a,0x0f,0x1f,0xff]
#CHECK: ah	%r15, 0                 # encoding: [0x4a,0xf0,0x00,0x00]

	ah	%r0, 0
	ah	%r0, 4095
	ah	%r0, 0(%r1)
	ah	%r0, 0(%r15)
	ah	%r0, 4095(%r1,%r15)
	ah	%r0, 4095(%r15,%r1)
	ah	%r15, 0
