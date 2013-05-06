# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: ear	%r0, 0
#CHECK: error: invalid register
#CHECK: ear	%r0, %r0
#CHECK: error: invalid register
#CHECK: ear	%a0, %r0

	ear	%r0, 0
	ear	%r0, %r0
	ear	%a0, %r0
