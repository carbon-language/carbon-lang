# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: lgr	%r16,%r1
#CHECK: error: invalid register
#CHECK: lgr	%f0,%r1
#CHECK: error: invalid register
#CHECK: lgr	%a0,%r1
#CHECK: error: invalid operand for instruction
#CHECK: lgr	%arid,%r1
#CHECK: error: invalid operand for instruction
#CHECK: lgr	%0,%r1
#CHECK: error: invalid operand for instruction
#CHECK: lgr	0,%r1
#CHECK: error: unknown token in expression
#CHECK: lgr	(%r0),%r1
#CHECK: error: unknown token in expression
#CHECK: lgr	%,%r1

	lgr	%r16,%r1
	lgr	%f0,%r1
	lgr	%a0,%r1
	lgr	%arid,%r1
	lgr	%0,%r1
	lgr	0,%r1
	lgr	(%r0),%r1
	lgr	%,%r1
