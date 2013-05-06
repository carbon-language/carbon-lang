# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: dsgr	%r0, %r0                # encoding: [0xb9,0x0d,0x00,0x00]
#CHECK: dsgr	%r0, %r15               # encoding: [0xb9,0x0d,0x00,0x0f]
#CHECK: dsgr	%r14, %r0               # encoding: [0xb9,0x0d,0x00,0xe0]
#CHECK: dsgr	%r6, %r9                # encoding: [0xb9,0x0d,0x00,0x69]

	dsgr	%r0,%r0
	dsgr	%r0,%r15
	dsgr	%r14,%r0
	dsgr	%r6,%r9
