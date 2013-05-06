# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cdgbr	%r0, %r0
#CHECK: error: invalid register
#CHECK: cdgbr	%f0, %f0
#CHECK: error: invalid register
#CHECK: cdgbr	%f0, %a0
#CHECK: error: invalid register
#CHECK: cdgbr	%a0, %r0

	cdgbr	%r0, %r0
	cdgbr	%f0, %f0
	cdgbr	%f0, %a0
	cdgbr	%a0, %r0
