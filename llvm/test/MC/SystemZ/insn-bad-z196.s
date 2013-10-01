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
#CHECK: aih	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: aih	%r0, (1 << 31)

	aih	%r0, (-1 << 31) - 1
	aih	%r0, (1 << 31)

#CHECK: error: invalid operand
#CHECK: chf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: chf	%r0, 524288

	chf	%r0, -524289
	chf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: cih	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: cih	%r0, (1 << 31)

	cih	%r0, (-1 << 31) - 1
	cih	%r0, (1 << 31)

#CHECK: error: invalid operand
#CHECK: clhf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: clhf	%r0, 524288

	clhf	%r0, -524289
	clhf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: clih	%r0, -1
#CHECK: error: invalid operand
#CHECK: clih	%r0, (1 << 32)

	clih	%r0, -1
	clih	%r0, (1 << 32)

#CHECK: error: invalid operand
#CHECK: fidbra	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: fidbra	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: fidbra	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: fidbra	%f0, 16, %f0, 0

	fidbra	%f0, 0, %f0, -1
	fidbra	%f0, 0, %f0, 16
	fidbra	%f0, -1, %f0, 0
	fidbra	%f0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: fiebra	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: fiebra	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: fiebra	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: fiebra	%f0, 16, %f0, 0

	fiebra	%f0, 0, %f0, -1
	fiebra	%f0, 0, %f0, 16
	fiebra	%f0, -1, %f0, 0
	fiebra	%f0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: fixbra	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: fixbra	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: fixbra	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: fixbra	%f0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: fixbra	%f0, 0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: fixbra	%f2, 0, %f0, 0

	fixbra	%f0, 0, %f0, -1
	fixbra	%f0, 0, %f0, 16
	fixbra	%f0, -1, %f0, 0
	fixbra	%f0, 16, %f0, 0
	fixbra	%f0, 0, %f2, 0
	fixbra	%f2, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: lbh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lbh	%r0, 524288

	lbh	%r0, -524289
	lbh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lfh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lfh	%r0, 524288

	lfh	%r0, -524289
	lfh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lhh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lhh	%r0, 524288

	lhh	%r0, -524289
	lhh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llch	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llch	%r0, 524288

	llch	%r0, -524289
	llch	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llhh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llhh	%r0, 524288

	llhh	%r0, -524289
	llhh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: loc	%r0,0,-1
#CHECK: error: invalid operand
#CHECK: loc	%r0,0,16
#CHECK: error: invalid operand
#CHECK: loc	%r0,-524289,1
#CHECK: error: invalid operand
#CHECK: loc	%r0,524288,1
#CHECK: error: invalid use of indexed addressing
#CHECK: loc	%r0,0(%r1,%r2),1

	loc	%r0,0,-1
	loc	%r0,0,16
	loc	%r0,-524289,1
	loc	%r0,524288,1
	loc	%r0,0(%r1,%r2),1

#CHECK: error: invalid operand
#CHECK: locg	%r0,0,-1
#CHECK: error: invalid operand
#CHECK: locg	%r0,0,16
#CHECK: error: invalid operand
#CHECK: locg	%r0,-524289,1
#CHECK: error: invalid operand
#CHECK: locg	%r0,524288,1
#CHECK: error: invalid use of indexed addressing
#CHECK: locg	%r0,0(%r1,%r2),1

	locg	%r0,0,-1
	locg	%r0,0,16
	locg	%r0,-524289,1
	locg	%r0,524288,1
	locg	%r0,0(%r1,%r2),1

#CHECK: error: invalid operand
#CHECK: locgr	%r0,%r0,-1
#CHECK: error: invalid operand
#CHECK: locgr	%r0,%r0,16

	locgr	%r0,%r0,-1
	locgr	%r0,%r0,16

#CHECK: error: invalid operand
#CHECK: locr	%r0,%r0,-1
#CHECK: error: invalid operand
#CHECK: locr	%r0,%r0,16

	locr	%r0,%r0,-1
	locr	%r0,%r0,16

#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,0,256,0
#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,256,0,0

	risbhg	%r0,%r0,0,0,-1
	risbhg	%r0,%r0,0,0,64
	risbhg	%r0,%r0,0,-1,0
	risbhg	%r0,%r0,0,256,0
	risbhg	%r0,%r0,-1,0,0
	risbhg	%r0,%r0,256,0,0

#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,0,256,0
#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,256,0,0

	risblg	%r0,%r0,0,0,-1
	risblg	%r0,%r0,0,0,64
	risblg	%r0,%r0,0,-1,0
	risblg	%r0,%r0,0,256,0
	risblg	%r0,%r0,-1,0,0
	risblg	%r0,%r0,256,0,0

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
#CHECK: stch	%r0, -524289
#CHECK: error: invalid operand
#CHECK: stch	%r0, 524288

	stch	%r0, -524289
	stch	%r0, 524288

#CHECK: error: invalid operand
#CHECK: sthh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: sthh	%r0, 524288

	sthh	%r0, -524289
	sthh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: stfh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: stfh	%r0, 524288

	stfh	%r0, -524289
	stfh	%r0, 524288

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
