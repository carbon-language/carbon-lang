# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: stc	%r0, 0                  # encoding: [0x42,0x00,0x00,0x00]
#CHECK: stc	%r0, 4095               # encoding: [0x42,0x00,0x0f,0xff]
#CHECK: stc	%r0, 0(%r1)             # encoding: [0x42,0x00,0x10,0x00]
#CHECK: stc	%r0, 0(%r15)            # encoding: [0x42,0x00,0xf0,0x00]
#CHECK: stc	%r0, 4095(%r1,%r15)     # encoding: [0x42,0x01,0xff,0xff]
#CHECK: stc	%r0, 4095(%r15,%r1)     # encoding: [0x42,0x0f,0x1f,0xff]
#CHECK: stc	%r15, 0                 # encoding: [0x42,0xf0,0x00,0x00]

	stc	%r0, 0
	stc	%r0, 4095
	stc	%r0, 0(%r1)
	stc	%r0, 0(%r15)
	stc	%r0, 4095(%r1,%r15)
	stc	%r0, 4095(%r15,%r1)
	stc	%r15, 0
