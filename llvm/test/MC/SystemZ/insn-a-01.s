# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: a	%r0, 0                  # encoding: [0x5a,0x00,0x00,0x00]
#CHECK: a	%r0, 4095               # encoding: [0x5a,0x00,0x0f,0xff]
#CHECK: a	%r0, 0(%r1)             # encoding: [0x5a,0x00,0x10,0x00]
#CHECK: a	%r0, 0(%r15)            # encoding: [0x5a,0x00,0xf0,0x00]
#CHECK: a	%r0, 4095(%r1,%r15)     # encoding: [0x5a,0x01,0xff,0xff]
#CHECK: a	%r0, 4095(%r15,%r1)     # encoding: [0x5a,0x0f,0x1f,0xff]
#CHECK: a	%r15, 0                 # encoding: [0x5a,0xf0,0x00,0x00]

	a	%r0, 0
	a	%r0, 4095
	a	%r0, 0(%r1)
	a	%r0, 0(%r15)
	a	%r0, 4095(%r1,%r15)
	a	%r0, 4095(%r15,%r1)
	a	%r15, 0
