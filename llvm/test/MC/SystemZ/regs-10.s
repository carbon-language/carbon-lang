# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

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
