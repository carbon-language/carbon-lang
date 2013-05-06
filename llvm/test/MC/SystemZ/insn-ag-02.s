# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: ag	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ag	%r0, 524288

	ag	%r0, -524289
	ag	%r0, 524288
