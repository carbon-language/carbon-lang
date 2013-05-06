# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: alcr	%r0, %r0                # encoding: [0xb9,0x98,0x00,0x00]
#CHECK: alcr	%r0, %r15               # encoding: [0xb9,0x98,0x00,0x0f]
#CHECK: alcr	%r15, %r0               # encoding: [0xb9,0x98,0x00,0xf0]
#CHECK: alcr	%r7, %r8                # encoding: [0xb9,0x98,0x00,0x78]

	alcr	%r0,%r0
	alcr	%r0,%r15
	alcr	%r15,%r0
	alcr	%r7,%r8
