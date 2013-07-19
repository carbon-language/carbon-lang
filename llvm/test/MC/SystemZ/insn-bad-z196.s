# For z196 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=z196 < %s 2> %t
# RUN: FileCheck < %t %s
#CHECK: error: invalid operand
#CHECK: sllk	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: sllk	%r0,%r0,524288
#CHECK: error: %r0 used in an address
#CHECK: sllk	%r0,%r0,0(%r0)
#CHECK: error: invalid use of indexed addressing
#CHECK: sllk	%r0,%r0,0(%r1,%r2)

	sllk	%r0,%r0,-524289
	sllk	%r0,%r0,524288
	sllk	%r0,%r0,0(%r0)
	sllk	%r0,%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: srak	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: srak	%r0,%r0,524288
#CHECK: error: %r0 used in an address
#CHECK: srak	%r0,%r0,0(%r0)
#CHECK: error: invalid use of indexed addressing
#CHECK: srak	%r0,%r0,0(%r1,%r2)

	srak	%r0,%r0,-524289
	srak	%r0,%r0,524288
	srak	%r0,%r0,0(%r0)
	srak	%r0,%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: srlk	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: srlk	%r0,%r0,524288
#CHECK: error: %r0 used in an address
#CHECK: srlk	%r0,%r0,0(%r0)
#CHECK: error: invalid use of indexed addressing
#CHECK: srlk	%r0,%r0,0(%r1,%r2)

	srlk	%r0,%r0,-524289
	srlk	%r0,%r0,524288
	srlk	%r0,%r0,0(%r0)
	srlk	%r0,%r0,0(%r1,%r2)
