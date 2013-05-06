# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: cgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cgf	%r0, 524288

	cgf	%r0, -524289
	cgf	%r0, 524288
