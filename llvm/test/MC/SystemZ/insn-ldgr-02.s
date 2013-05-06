# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: ldgr	%f0, %f0
#CHECK: error: invalid register
#CHECK: ldgr	%r0, %r0
#CHECK: error: invalid register
#CHECK: ldgr	%f0, %a0
#CHECK: error: invalid register
#CHECK: ldgr	%a0, %r0

	ldgr	%f0, %f0
	ldgr	%r0, %r0
	ldgr	%f0, %a0
	ldgr	%a0, %r0
