# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: csy	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: csy	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: csy	%r0, %r0, 0(%r1,%r2)

	csy	%r0, %r0, -524289
	csy	%r0, %r0, 524288
	csy	%r0, %r0, 0(%r1,%r2)
