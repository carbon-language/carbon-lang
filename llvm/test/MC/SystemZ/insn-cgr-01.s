# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cgr	%r0, %r0                # encoding: [0xb9,0x20,0x00,0x00]
#CHECK: cgr	%r0, %r15               # encoding: [0xb9,0x20,0x00,0x0f]
#CHECK: cgr	%r15, %r0               # encoding: [0xb9,0x20,0x00,0xf0]
#CHECK: cgr	%r7, %r8                # encoding: [0xb9,0x20,0x00,0x78]

	cgr	%r0,%r0
	cgr	%r0,%r15
	cgr	%r15,%r0
	cgr	%r7,%r8
