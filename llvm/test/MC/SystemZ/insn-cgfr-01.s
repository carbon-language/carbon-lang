# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cgfr	%r0, %r0                # encoding: [0xb9,0x30,0x00,0x00]
#CHECK: cgfr	%r0, %r15               # encoding: [0xb9,0x30,0x00,0x0f]
#CHECK: cgfr	%r15, %r0               # encoding: [0xb9,0x30,0x00,0xf0]
#CHECK: cgfr	%r7, %r8                # encoding: [0xb9,0x30,0x00,0x78]

	cgfr	%r0,%r0
	cgfr	%r0,%r15
	cgfr	%r15,%r0
	cgfr	%r7,%r8
