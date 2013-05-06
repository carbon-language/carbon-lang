# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: srl	%r0, 0                  # encoding: [0x88,0x00,0x00,0x00]
#CHECK: srl	%r7, 0                  # encoding: [0x88,0x70,0x00,0x00]
#CHECK: srl	%r15, 0                 # encoding: [0x88,0xf0,0x00,0x00]
#CHECK: srl	%r0, 4095               # encoding: [0x88,0x00,0x0f,0xff]
#CHECK: srl	%r0, 0(%r1)             # encoding: [0x88,0x00,0x10,0x00]
#CHECK: srl	%r0, 0(%r15)            # encoding: [0x88,0x00,0xf0,0x00]
#CHECK: srl	%r0, 4095(%r1)          # encoding: [0x88,0x00,0x1f,0xff]
#CHECK: srl	%r0, 4095(%r15)         # encoding: [0x88,0x00,0xff,0xff]

	srl	%r0,0
	srl	%r7,0
	srl	%r15,0
	srl	%r0,4095
	srl	%r0,0(%r1)
	srl	%r0,0(%r15)
	srl	%r0,4095(%r1)
	srl	%r0,4095(%r15)
