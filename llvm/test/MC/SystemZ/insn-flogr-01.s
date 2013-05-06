# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: flogr	%r0, %r0                # encoding: [0xb9,0x83,0x00,0x00]
#CHECK: flogr	%r0, %r15               # encoding: [0xb9,0x83,0x00,0x0f]
#CHECK: flogr	%r10, %r9               # encoding: [0xb9,0x83,0x00,0xa9]
#CHECK: flogr	%r14, %r0               # encoding: [0xb9,0x83,0x00,0xe0]

	flogr	%r0, %r0
	flogr	%r0, %r15
	flogr	%r10, %r9
	flogr	%r14, %r0
