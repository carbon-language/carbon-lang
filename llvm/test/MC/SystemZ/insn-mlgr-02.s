# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: mlgr	%r1, %r0
#CHECK: error: invalid register
#CHECK: mlgr	%r15, %r0

	mlgr	%r1, %r0
	mlgr	%r15, %r0
