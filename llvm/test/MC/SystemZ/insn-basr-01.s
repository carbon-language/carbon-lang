# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: basr	%r0, %r1                # encoding: [0x0d,0x01]
#CHECK: basr	%r0, %r15               # encoding: [0x0d,0x0f]
#CHECK: basr	%r14, %r9               # encoding: [0x0d,0xe9]
#CHECK: basr	%r15, %r1               # encoding: [0x0d,0xf1]

	basr	%r0,%r1
	basr	%r0,%r15
	basr	%r14,%r9
	basr	%r15,%r1

