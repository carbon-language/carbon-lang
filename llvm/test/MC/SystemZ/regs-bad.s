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

#CHECK: error: invalid register
#CHECK: ler	%f1,%f16
#CHECK: error: invalid register
#CHECK: ler	%f1,%r0
#CHECK: error: invalid register
#CHECK: ler	%f1,%a0
#CHECK: error: invalid operand for instruction
#CHECK: ler	%f1,%fly
#CHECK: error: invalid operand for instruction
#CHECK: ler	%f1,%0
#CHECK: error: invalid operand for instruction
#CHECK: ler	%f1,0
#CHECK: error: unknown token in expression
#CHECK: ler	%f1,(%f0)
#CHECK: error: unknown token in expression
#CHECK: ler	%f1,%

	ler	%f1,%f16
	ler	%f1,%r0
	ler	%f1,%a0
	ler	%f1,%fly
	ler	%f1,%0
	ler	%f1,0
	ler	%f1,(%f0)
	ler	%f1,%

#CHECK: error: invalid register
#CHECK: ldr	%f1,%f16
#CHECK: error: invalid register
#CHECK: ldr	%f1,%r0
#CHECK: error: invalid register
#CHECK: ldr	%f1,%a0
#CHECK: error: invalid operand for instruction
#CHECK: ldr	%f1,%fly
#CHECK: error: invalid operand for instruction
#CHECK: ldr	%f1,%0
#CHECK: error: invalid operand for instruction
#CHECK: ldr	%f1,0
#CHECK: error: unknown token in expression
#CHECK: ldr	%f1,(%f0)
#CHECK: error: unknown token in expression
#CHECK: ldr	%f1,%

	ldr	%f1,%f16
	ldr	%f1,%r0
	ldr	%f1,%a0
	ldr	%f1,%fly
	ldr	%f1,%0
	ldr	%f1,0
	ldr	%f1,(%f0)
	ldr	%f1,%

#CHECK: error: invalid register
#CHECK: lxr	%f1,%f2
#CHECK: error: invalid register
#CHECK: lxr	%f1,%f16
#CHECK: error: invalid register
#CHECK: lxr	%f1,%r0
#CHECK: error: invalid register
#CHECK: lxr	%f1,%a0
#CHECK: error: invalid operand for instruction
#CHECK: lxr	%f1,%fly
#CHECK: error: invalid operand for instruction
#CHECK: lxr	%f1,%0
#CHECK: error: invalid operand for instruction
#CHECK: lxr	%f1,0
#CHECK: error: unknown token in expression
#CHECK: lxr	%f1,(%f0)
#CHECK: error: unknown token in expression
#CHECK: lxr	%f1,%

	lxr	%f1,%f2
	lxr	%f1,%f16
	lxr	%f1,%r0
	lxr	%f1,%a0
	lxr	%f1,%fly
	lxr	%f1,%0
	lxr	%f1,0
	lxr	%f1,(%f0)
	lxr	%f1,%

#CHECK: error: invalid register
#CHECK: .cfi_offset %a0,0
#CHECK: error: register expected
#CHECK: .cfi_offset %foo,0
#CHECK: error: register expected
#CHECK: .cfi_offset %,0
#CHECK: error: register expected
#CHECK: .cfi_offset r0,0

	.cfi_startproc
	.cfi_offset %a0,0
	.cfi_offset %foo,0
	.cfi_offset %,0
	.cfi_offset r0,0
	.cfi_endproc

#CHECK: error: %r0 used in an address
#CHECK: sll	%r2,8(%r0)
#CHECK: error: %r0 used in an address
#CHECK: br	%r0
#CHECK: error: %r0 used in an address
#CHECK: l	%r1,8(%r0)
#CHECK: error: %r0 used in an address
#CHECK: l	%r1,8(%r0,%r15)
#CHECK: error: %r0 used in an address
#CHECK: l	%r1,8(%r15,%r0)

	sll	%r2,8(%r0)
	br	%r0
	l	%r1,8(%r0)
	l	%r1,8(%r0,%r15)
	l	%r1,8(%r15,%r0)
