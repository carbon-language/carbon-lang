# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,0,64,0
#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,64,0,0

	risbg	%r0,%r0,0,0,-1
	risbg	%r0,%r0,0,0,64
	risbg	%r0,%r0,0,-1,0
	risbg	%r0,%r0,0,64,0
	risbg	%r0,%r0,-1,0,0
	risbg	%r0,%r0,64,0,0
