# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: dlr	%r1,%r8
#CHECK: error: invalid register
#CHECK: dlr	%r16,%r1
#CHECK: error: invalid register
#CHECK: dlr	%f0,%r1
#CHECK: error: invalid register
#CHECK: dlr	%a0,%r1
#CHECK: error: invalid operand for instruction
#CHECK: dlr	%arid,%r1
#CHECK: error: invalid operand for instruction
#CHECK: dlr	%0,%r1
#CHECK: error: invalid operand for instruction
#CHECK: dlr	0,%r1
#CHECK: error: unknown token in expression
#CHECK: dlr	(%r0),%r1
#CHECK: error: unknown token in expression
#CHECK: dlr	%,%r1

	dlr	%r1,%r8
	dlr	%r16,%r1
	dlr	%f0,%r1
	dlr	%a0,%r1
	dlr	%arid,%r1
	dlr	%0,%r1
	dlr	0,%r1
	dlr	(%r0),%r1
	dlr	%,%r1
