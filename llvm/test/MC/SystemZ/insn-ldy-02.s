# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: ldy	%f0, -524289
#CHECK: error: invalid operand
#CHECK: ldy	%f0, 524288

	ldy	%f0, -524289
	ldy	%f0, 524288
