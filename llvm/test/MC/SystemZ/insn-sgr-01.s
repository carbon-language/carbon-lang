# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sgr	%r0, %r0                # encoding: [0xb9,0x09,0x00,0x00]
#CHECK: sgr	%r0, %r15               # encoding: [0xb9,0x09,0x00,0x0f]
#CHECK: sgr	%r15, %r0               # encoding: [0xb9,0x09,0x00,0xf0]
#CHECK: sgr	%r7, %r8                # encoding: [0xb9,0x09,0x00,0x78]

	sgr	%r0,%r0
	sgr	%r0,%r15
	sgr	%r15,%r0
	sgr	%r7,%r8
