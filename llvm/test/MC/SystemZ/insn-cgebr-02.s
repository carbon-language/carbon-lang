# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cgebr	%r0, 0, %r0
#CHECK: error: invalid register
#CHECK: cgebr	%f0, 0, %f0
#CHECK: error: invalid operand
#CHECK: cgebr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cgebr	%r0, 16, %f0

	cgebr	%r0, 0, %r0
	cgebr	%f0, 0, %f0
	cgebr	%r0, -1, %f0
	cgebr	%r0, 16, %f0
