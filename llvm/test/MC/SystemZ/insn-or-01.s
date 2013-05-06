# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: or	%r0, %r0                # encoding: [0x16,0x00]
#CHECK: or	%r0, %r15               # encoding: [0x16,0x0f]
#CHECK: or	%r15, %r0               # encoding: [0x16,0xf0]
#CHECK: or	%r7, %r8                # encoding: [0x16,0x78]

	or	%r0,%r0
	or	%r0,%r15
	or	%r15,%r0
	or	%r7,%r8
