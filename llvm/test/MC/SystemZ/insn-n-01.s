# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: n	%r0, 0                  # encoding: [0x54,0x00,0x00,0x00]
#CHECK: n	%r0, 4095               # encoding: [0x54,0x00,0x0f,0xff]
#CHECK: n	%r0, 0(%r1)             # encoding: [0x54,0x00,0x10,0x00]
#CHECK: n	%r0, 0(%r15)            # encoding: [0x54,0x00,0xf0,0x00]
#CHECK: n	%r0, 4095(%r1,%r15)     # encoding: [0x54,0x01,0xff,0xff]
#CHECK: n	%r0, 4095(%r15,%r1)     # encoding: [0x54,0x0f,0x1f,0xff]
#CHECK: n	%r15, 0                 # encoding: [0x54,0xf0,0x00,0x00]

	n	%r0, 0
	n	%r0, 4095
	n	%r0, 0(%r1)
	n	%r0, 0(%r15)
	n	%r0, 4095(%r1,%r15)
	n	%r0, 4095(%r15,%r1)
	n	%r15, 0
