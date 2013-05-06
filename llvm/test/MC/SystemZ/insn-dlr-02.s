# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: dlr	%r1, %r0
#CHECK: error: invalid register
#CHECK: dlr	%r15, %r0

	dlr	%r1, %r0
	dlr	%r15, %r0
