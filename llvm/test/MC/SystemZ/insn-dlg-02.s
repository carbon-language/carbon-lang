# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: dlg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: dlg	%r0, 524288
#CHECK: error: invalid register
#CHECK: dlg	%r1, 0
#CHECK: error: invalid register
#CHECK: dlg	%r15, 0

	dlg	%r0, -524289
	dlg	%r0, 524288
	dlg	%r1, 0
	dlg	%r15, 0
