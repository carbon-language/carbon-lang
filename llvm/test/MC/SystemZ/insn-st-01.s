# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: st	%r0, 0                  # encoding: [0x50,0x00,0x00,0x00]
#CHECK: st	%r0, 4095               # encoding: [0x50,0x00,0x0f,0xff]
#CHECK: st	%r0, 0(%r1)             # encoding: [0x50,0x00,0x10,0x00]
#CHECK: st	%r0, 0(%r15)            # encoding: [0x50,0x00,0xf0,0x00]
#CHECK: st	%r0, 4095(%r1,%r15)     # encoding: [0x50,0x01,0xff,0xff]
#CHECK: st	%r0, 4095(%r15,%r1)     # encoding: [0x50,0x0f,0x1f,0xff]
#CHECK: st	%r15, 0                 # encoding: [0x50,0xf0,0x00,0x00]

	st	%r0, 0
	st	%r0, 4095
	st	%r0, 0(%r1)
	st	%r0, 0(%r15)
	st	%r0, 4095(%r1,%r15)
	st	%r0, 4095(%r15,%r1)
	st	%r15, 0
