# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lr	%r0, %r1                # encoding: [0x18,0x01]
#CHECK: lr	%r2, %r3                # encoding: [0x18,0x23]
#CHECK: lr	%r4, %r5                # encoding: [0x18,0x45]
#CHECK: lr	%r6, %r7                # encoding: [0x18,0x67]
#CHECK: lr	%r8, %r9                # encoding: [0x18,0x89]
#CHECK: lr	%r10, %r11              # encoding: [0x18,0xab]
#CHECK: lr	%r12, %r13              # encoding: [0x18,0xcd]
#CHECK: lr	%r14, %r15              # encoding: [0x18,0xef]

	lr	%r0,%r1
	lr	%r2,%r3
	lr	%r4,%r5
	lr	%r6,%r7
	lr	%r8,%r9
	lr	%r10,%r11
	lr	%r12,%r13
	lr	%r14,%r15
