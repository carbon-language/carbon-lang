# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sll	%r0, 0                  # encoding: [0x89,0x00,0x00,0x00]
#CHECK: sll	%r7, 0                  # encoding: [0x89,0x70,0x00,0x00]
#CHECK: sll	%r15, 0                 # encoding: [0x89,0xf0,0x00,0x00]
#CHECK: sll	%r0, 4095               # encoding: [0x89,0x00,0x0f,0xff]
#CHECK: sll	%r0, 0(%r1)             # encoding: [0x89,0x00,0x10,0x00]
#CHECK: sll	%r0, 0(%r15)            # encoding: [0x89,0x00,0xf0,0x00]
#CHECK: sll	%r0, 4095(%r1)          # encoding: [0x89,0x00,0x1f,0xff]
#CHECK: sll	%r0, 4095(%r15)         # encoding: [0x89,0x00,0xff,0xff]

	sll	%r0,0
	sll	%r7,0
	sll	%r15,0
	sll	%r0,4095
	sll	%r0,0(%r1)
	sll	%r0,0(%r15)
	sll	%r0,4095(%r1)
	sll	%r0,4095(%r15)
