# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: dlgr	%r0, %r0                # encoding: [0xb9,0x87,0x00,0x00]
#CHECK: dlgr	%r0, %r15               # encoding: [0xb9,0x87,0x00,0x0f]
#CHECK: dlgr	%r14, %r0               # encoding: [0xb9,0x87,0x00,0xe0]
#CHECK: dlgr	%r6, %r9                # encoding: [0xb9,0x87,0x00,0x69]

	dlgr	%r0,%r0
	dlgr	%r0,%r15
	dlgr	%r14,%r0
	dlgr	%r6,%r9
