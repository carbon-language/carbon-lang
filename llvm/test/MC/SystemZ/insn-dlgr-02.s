# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: dlgr	%r1, %r0
#CHECK: error: invalid register
#CHECK: dlgr	%r15, %r0

	dlgr	%r1, %r0
	dlgr	%r15, %r0
