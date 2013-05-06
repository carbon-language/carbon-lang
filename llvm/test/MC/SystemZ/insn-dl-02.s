# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: dl	%r0, -524289
#CHECK: error: invalid operand
#CHECK: dl	%r0, 524288
#CHECK: error: invalid register
#CHECK: dl	%r1, 0
#CHECK: error: invalid register
#CHECK: dl	%r15, 0

	dl	%r0, -524289
	dl	%r0, 524288
	dl	%r1, 0
	dl	%r15, 0
