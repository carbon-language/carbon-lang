# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: xr	%r0, %r0                # encoding: [0x17,0x00]
#CHECK: xr	%r0, %r15               # encoding: [0x17,0x0f]
#CHECK: xr	%r15, %r0               # encoding: [0x17,0xf0]
#CHECK: xr	%r7, %r8                # encoding: [0x17,0x78]

	xr	%r0,%r0
	xr	%r0,%r15
	xr	%r15,%r0
	xr	%r7,%r8
