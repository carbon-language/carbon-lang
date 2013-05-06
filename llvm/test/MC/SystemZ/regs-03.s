# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: dlr	%r0, %r0                # encoding: [0xb9,0x97,0x00,0x00]
#CHECK: dlr	%r2, %r0                # encoding: [0xb9,0x97,0x00,0x20]
#CHECK: dlr	%r4, %r0                # encoding: [0xb9,0x97,0x00,0x40]
#CHECK: dlr	%r6, %r0                # encoding: [0xb9,0x97,0x00,0x60]
#CHECK: dlr	%r8, %r0                # encoding: [0xb9,0x97,0x00,0x80]
#CHECK: dlr	%r10, %r0               # encoding: [0xb9,0x97,0x00,0xa0]
#CHECK: dlr	%r12, %r0               # encoding: [0xb9,0x97,0x00,0xc0]
#CHECK: dlr	%r14, %r0               # encoding: [0xb9,0x97,0x00,0xe0]

	dlr	%r0,%r0
	dlr	%r2,%r0
	dlr	%r4,%r0
	dlr	%r6,%r0
	dlr	%r8,%r0
	dlr	%r10,%r0
	dlr	%r12,%r0
	dlr	%r14,%r0
