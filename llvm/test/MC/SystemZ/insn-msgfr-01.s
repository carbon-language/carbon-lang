# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: msgfr	%r0, %r0                # encoding: [0xb9,0x1c,0x00,0x00]
#CHECK: msgfr	%r0, %r15               # encoding: [0xb9,0x1c,0x00,0x0f]
#CHECK: msgfr	%r15, %r0               # encoding: [0xb9,0x1c,0x00,0xf0]
#CHECK: msgfr	%r7, %r8                # encoding: [0xb9,0x1c,0x00,0x78]

	msgfr	%r0,%r0
	msgfr	%r0,%r15
	msgfr	%r15,%r0
	msgfr	%r7,%r8
