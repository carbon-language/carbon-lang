# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sgfr	%r0, %r0                # encoding: [0xb9,0x19,0x00,0x00]
#CHECK: sgfr	%r0, %r15               # encoding: [0xb9,0x19,0x00,0x0f]
#CHECK: sgfr	%r15, %r0               # encoding: [0xb9,0x19,0x00,0xf0]
#CHECK: sgfr	%r7, %r8                # encoding: [0xb9,0x19,0x00,0x78]

	sgfr	%r0,%r0
	sgfr	%r0,%r15
	sgfr	%r15,%r0
	sgfr	%r7,%r8
