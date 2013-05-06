# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: stey	%f0, -524289
#CHECK: error: invalid operand
#CHECK: stey	%f0, 524288

	stey	%f0, -524289
	stey	%f0, 524288
