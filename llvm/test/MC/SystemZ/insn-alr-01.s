# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: alr	%r0, %r0                # encoding: [0x1e,0x00]
#CHECK: alr	%r0, %r15               # encoding: [0x1e,0x0f]
#CHECK: alr	%r15, %r0               # encoding: [0x1e,0xf0]
#CHECK: alr	%r7, %r8                # encoding: [0x1e,0x78]

	alr	%r0,%r0
	alr	%r0,%r15
	alr	%r15,%r0
	alr	%r7,%r8
