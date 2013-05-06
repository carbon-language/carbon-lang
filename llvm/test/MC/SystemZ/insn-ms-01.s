# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ms	%r0, 0                  # encoding: [0x71,0x00,0x00,0x00]
#CHECK: ms	%r0, 4095               # encoding: [0x71,0x00,0x0f,0xff]
#CHECK: ms	%r0, 0(%r1)             # encoding: [0x71,0x00,0x10,0x00]
#CHECK: ms	%r0, 0(%r15)            # encoding: [0x71,0x00,0xf0,0x00]
#CHECK: ms	%r0, 4095(%r1,%r15)     # encoding: [0x71,0x01,0xff,0xff]
#CHECK: ms	%r0, 4095(%r15,%r1)     # encoding: [0x71,0x0f,0x1f,0xff]
#CHECK: ms	%r15, 0                 # encoding: [0x71,0xf0,0x00,0x00]

	ms	%r0, 0
	ms	%r0, 4095
	ms	%r0, 0(%r1)
	ms	%r0, 0(%r15)
	ms	%r0, 4095(%r1,%r15)
	ms	%r0, 4095(%r15,%r1)
	ms	%r15, 0
