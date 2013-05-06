# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: dsgfr	%r0, %r0                # encoding: [0xb9,0x1d,0x00,0x00]
#CHECK: dsgfr	%r0, %r15               # encoding: [0xb9,0x1d,0x00,0x0f]
#CHECK: dsgfr	%r14, %r0               # encoding: [0xb9,0x1d,0x00,0xe0]
#CHECK: dsgfr	%r6, %r9                # encoding: [0xb9,0x1d,0x00,0x69]

	dsgfr	%r0,%r0
	dsgfr	%r0,%r15
	dsgfr	%r14,%r0
	dsgfr	%r6,%r9
