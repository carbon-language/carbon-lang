# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: lmg	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: lmg	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lmg	%r0, %r0, 0(%r1,%r2)

	lmg	%r0, %r0, -524289
	lmg	%r0, %r0, 524288
	lmg	%r0, %r0, 0(%r1,%r2)
