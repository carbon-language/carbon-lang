# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: l	%r0, 0                  # encoding: [0x58,0x00,0x00,0x00]
#CHECK: l	%r0, 4095               # encoding: [0x58,0x00,0x0f,0xff]
#CHECK: l	%r0, 0(%r1)             # encoding: [0x58,0x00,0x10,0x00]
#CHECK: l	%r0, 0(%r15)            # encoding: [0x58,0x00,0xf0,0x00]
#CHECK: l	%r0, 4095(%r1,%r15)     # encoding: [0x58,0x01,0xff,0xff]
#CHECK: l	%r0, 4095(%r15,%r1)     # encoding: [0x58,0x0f,0x1f,0xff]
#CHECK: l	%r15, 0                 # encoding: [0x58,0xf0,0x00,0x00]

	l	%r0, 0
	l	%r0, 4095
	l	%r0, 0(%r1)
	l	%r0, 0(%r15)
	l	%r0, 4095(%r1,%r15)
	l	%r0, 4095(%r15,%r1)
	l	%r15, 0
