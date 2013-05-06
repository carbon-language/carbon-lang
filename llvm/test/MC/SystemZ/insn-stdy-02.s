# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: stdy	%f0, -524289
#CHECK: error: invalid operand
#CHECK: stdy	%f0, 524288

	stdy	%f0, -524289
	stdy	%f0, 524288
