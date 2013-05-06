# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: lr	%r16,%r1
#CHECK: error: invalid register
#CHECK: lr	%f0,%r1
#CHECK: error: invalid register
#CHECK: lr	%a0,%r1
#CHECK: error: invalid operand for instruction
#CHECK: lr	%arid,%r1
#CHECK: error: invalid operand for instruction
#CHECK: lr	%0,%r1
#CHECK: error: invalid operand for instruction
#CHECK: lr	0,%r1
#CHECK: error: unknown token in expression
#CHECK: lr	(%r0),%r1
#CHECK: error: unknown token in expression
#CHECK: lr	%,%r1

	lr	%r16,%r1
	lr	%f0,%r1
	lr	%a0,%r1
	lr	%arid,%r1
	lr	%0,%r1
	lr	0,%r1
	lr	(%r0),%r1
	lr	%,%r1
