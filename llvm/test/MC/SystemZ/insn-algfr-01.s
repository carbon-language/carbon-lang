# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: algfr	%r0, %r0                # encoding: [0xb9,0x1a,0x00,0x00]
#CHECK: algfr	%r0, %r15               # encoding: [0xb9,0x1a,0x00,0x0f]
#CHECK: algfr	%r15, %r0               # encoding: [0xb9,0x1a,0x00,0xf0]
#CHECK: algfr	%r7, %r8                # encoding: [0xb9,0x1a,0x00,0x78]

	algfr	%r0,%r0
	algfr	%r0,%r15
	algfr	%r15,%r0
	algfr	%r7,%r8
