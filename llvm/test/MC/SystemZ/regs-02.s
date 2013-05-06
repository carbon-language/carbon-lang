# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lgr	%r0, %r1                # encoding: [0xb9,0x04,0x00,0x01]
#CHECK: lgr	%r2, %r3                # encoding: [0xb9,0x04,0x00,0x23]
#CHECK: lgr	%r4, %r5                # encoding: [0xb9,0x04,0x00,0x45]
#CHECK: lgr	%r6, %r7                # encoding: [0xb9,0x04,0x00,0x67]
#CHECK: lgr	%r8, %r9                # encoding: [0xb9,0x04,0x00,0x89]
#CHECK: lgr	%r10, %r11              # encoding: [0xb9,0x04,0x00,0xab]
#CHECK: lgr	%r12, %r13              # encoding: [0xb9,0x04,0x00,0xcd]
#CHECK: lgr	%r14, %r15              # encoding: [0xb9,0x04,0x00,0xef]

	lgr	%r0,%r1
	lgr	%r2,%r3
	lgr	%r4,%r5
	lgr	%r6,%r7
	lgr	%r8,%r9
	lgr	%r10,%r11
	lgr	%r12,%r13
	lgr	%r14,%r15
