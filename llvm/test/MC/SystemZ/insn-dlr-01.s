# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: dlr	%r0, %r0                # encoding: [0xb9,0x97,0x00,0x00]
#CHECK: dlr	%r0, %r15               # encoding: [0xb9,0x97,0x00,0x0f]
#CHECK: dlr	%r14, %r0               # encoding: [0xb9,0x97,0x00,0xe0]
#CHECK: dlr	%r6, %r9                # encoding: [0xb9,0x97,0x00,0x69]

	dlr	%r0,%r0
	dlr	%r0,%r15
	dlr	%r14,%r0
	dlr	%r6,%r9
