# For z196 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=z196 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: aghik	%r0, %r1, -32769
#CHECK: error: invalid operand
#CHECK: aghik	%r0, %r1, 32768
#CHECK: error: invalid operand
#CHECK: aghik	%r0, %r1, foo

	aghik	%r0, %r1, -32769
	aghik	%r0, %r1, 32768
	aghik	%r0, %r1, foo

#CHECK: error: invalid operand
#CHECK: ahik	%r0, %r1, -32769
#CHECK: error: invalid operand
#CHECK: ahik	%r0, %r1, 32768
#CHECK: error: invalid operand
#CHECK: ahik	%r0, %r1, foo

	ahik	%r0, %r1, -32769
	ahik	%r0, %r1, 32768
	ahik	%r0, %r1, foo

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

#CHECK: error: invalid operand
#CHECK: stoc	%r0,0,-1
#CHECK: error: invalid operand
#CHECK: stoc	%r0,0,16
#CHECK: error: invalid operand
#CHECK: stoc	%r0,-524289,1
#CHECK: error: invalid operand
#CHECK: stoc	%r0,524288,1
#CHECK: error: invalid use of indexed addressing
#CHECK: stoc	%r0,0(%r1,%r2),1

	stoc	%r0,0,-1
	stoc	%r0,0,16
	stoc	%r0,-524289,1
	stoc	%r0,524288,1
	stoc	%r0,0(%r1,%r2),1

#CHECK: error: invalid operand
#CHECK: stocg	%r0,0,-1
#CHECK: error: invalid operand
#CHECK: stocg	%r0,0,16
#CHECK: error: invalid operand
#CHECK: stocg	%r0,-524289,1
#CHECK: error: invalid operand
#CHECK: stocg	%r0,524288,1
#CHECK: error: invalid use of indexed addressing
#CHECK: stocg	%r0,0(%r1,%r2),1

	stocg	%r0,0,-1
	stocg	%r0,0,16
	stocg	%r0,-524289,1
	stocg	%r0,524288,1
	stocg	%r0,0(%r1,%r2),1
