# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

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
