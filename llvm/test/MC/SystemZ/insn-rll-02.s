# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: rll	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: rll	%r0,%r0,524288
#CHECK: error: %r0 used in an address
#CHECK: rll	%r0,%r0,0(%r0)
#CHECK: error: invalid use of indexed addressing
#CHECK: rll	%r0,%r0,0(%r1,%r2)

	rll	%r0,%r0,-524289
	rll	%r0,%r0,524288
	rll	%r0,%r0,0(%r0)
	rll	%r0,%r0,0(%r1,%r2)
