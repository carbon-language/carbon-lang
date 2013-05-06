# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: sllg	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: sllg	%r0,%r0,524288
#CHECK: error: %r0 used in an address
#CHECK: sllg	%r0,%r0,0(%r0)
#CHECK: error: invalid use of indexed addressing
#CHECK: sllg	%r0,%r0,0(%r1,%r2)

	sllg	%r0,%r0,-524289
	sllg	%r0,%r0,524288
	sllg	%r0,%r0,0(%r0)
	sllg	%r0,%r0,0(%r1,%r2)
