# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sra	%r0, 0                  # encoding: [0x8a,0x00,0x00,0x00]
#CHECK: sra	%r7, 0                  # encoding: [0x8a,0x70,0x00,0x00]
#CHECK: sra	%r15, 0                 # encoding: [0x8a,0xf0,0x00,0x00]
#CHECK: sra	%r0, 4095               # encoding: [0x8a,0x00,0x0f,0xff]
#CHECK: sra	%r0, 0(%r1)             # encoding: [0x8a,0x00,0x10,0x00]
#CHECK: sra	%r0, 0(%r15)            # encoding: [0x8a,0x00,0xf0,0x00]
#CHECK: sra	%r0, 4095(%r1)          # encoding: [0x8a,0x00,0x1f,0xff]
#CHECK: sra	%r0, 4095(%r15)         # encoding: [0x8a,0x00,0xff,0xff]

	sra	%r0,0
	sra	%r7,0
	sra	%r15,0
	sra	%r0,4095
	sra	%r0,0(%r1)
	sra	%r0,0(%r15)
	sra	%r0,4095(%r1)
	sra	%r0,4095(%r15)
