# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

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
