# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: s	%r0, 0                  # encoding: [0x5b,0x00,0x00,0x00]
#CHECK: s	%r0, 4095               # encoding: [0x5b,0x00,0x0f,0xff]
#CHECK: s	%r0, 0(%r1)             # encoding: [0x5b,0x00,0x10,0x00]
#CHECK: s	%r0, 0(%r15)            # encoding: [0x5b,0x00,0xf0,0x00]
#CHECK: s	%r0, 4095(%r1,%r15)     # encoding: [0x5b,0x01,0xff,0xff]
#CHECK: s	%r0, 4095(%r15,%r1)     # encoding: [0x5b,0x0f,0x1f,0xff]
#CHECK: s	%r15, 0                 # encoding: [0x5b,0xf0,0x00,0x00]

	s	%r0, 0
	s	%r0, 4095
	s	%r0, 0(%r1)
	s	%r0, 0(%r15)
	s	%r0, 4095(%r1,%r15)
	s	%r0, 4095(%r15,%r1)
	s	%r15, 0
