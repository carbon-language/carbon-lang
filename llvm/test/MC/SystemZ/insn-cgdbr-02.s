# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cgdbr	%r0, 0, %r0
#CHECK: error: invalid register
#CHECK: cgdbr	%f0, 0, %f0
#CHECK: error: invalid operand
#CHECK: cgdbr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cgdbr	%r0, 16, %f0

	cgdbr	%r0, 0, %r0
	cgdbr	%f0, 0, %f0
	cgdbr	%r0, -1, %f0
	cgdbr	%r0, 16, %f0
