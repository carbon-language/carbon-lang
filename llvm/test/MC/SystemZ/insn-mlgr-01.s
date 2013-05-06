# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: mlgr	%r0, %r0                # encoding: [0xb9,0x86,0x00,0x00]
#CHECK: mlgr	%r0, %r15               # encoding: [0xb9,0x86,0x00,0x0f]
#CHECK: mlgr	%r14, %r0               # encoding: [0xb9,0x86,0x00,0xe0]
#CHECK: mlgr	%r6, %r9                # encoding: [0xb9,0x86,0x00,0x69]

	mlgr	%r0,%r0
	mlgr	%r0,%r15
	mlgr	%r14,%r0
	mlgr	%r6,%r9
