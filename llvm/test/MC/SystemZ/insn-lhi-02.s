# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: lhi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: lhi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: lhi	%r0, foo

	lhi	%r0, -32769
	lhi	%r0, 32768
	lhi	%r0, foo
