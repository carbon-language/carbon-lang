# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sth	%r0, 0                  # encoding: [0x40,0x00,0x00,0x00]
#CHECK: sth	%r0, 4095               # encoding: [0x40,0x00,0x0f,0xff]
#CHECK: sth	%r0, 0(%r1)             # encoding: [0x40,0x00,0x10,0x00]
#CHECK: sth	%r0, 0(%r15)            # encoding: [0x40,0x00,0xf0,0x00]
#CHECK: sth	%r0, 4095(%r1,%r15)     # encoding: [0x40,0x01,0xff,0xff]
#CHECK: sth	%r0, 4095(%r15,%r1)     # encoding: [0x40,0x0f,0x1f,0xff]
#CHECK: sth	%r15, 0                 # encoding: [0x40,0xf0,0x00,0x00]

	sth	%r0, 0
	sth	%r0, 4095
	sth	%r0, 0(%r1)
	sth	%r0, 0(%r15)
	sth	%r0, 4095(%r1,%r15)
	sth	%r0, 4095(%r15,%r1)
	sth	%r15, 0
