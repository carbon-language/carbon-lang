# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: cg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cg	%r0, 524288

	cg	%r0, -524289
	cg	%r0, 524288
