# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cefbr	%r0, %r0
#CHECK: error: invalid register
#CHECK: cefbr	%f0, %f0
#CHECK: error: invalid register
#CHECK: cefbr	%f0, %a0
#CHECK: error: invalid register
#CHECK: cefbr	%a0, %r0

	cefbr	%r0, %r0
	cefbr	%f0, %f0
	cefbr	%f0, %a0
	cefbr	%a0, %r0
