# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: sll	%r0,-1
#CHECK: error: invalid operand
#CHECK: sll	%r0,4096
#CHECK: error: %r0 used in an address
#CHECK: sll	%r0,0(%r0)
#CHECK: error: invalid use of indexed addressing
#CHECK: sll	%r0,0(%r1,%r2)

	sll	%r0,-1
	sll	%r0,4096
	sll	%r0,0(%r0)
	sll	%r0,0(%r1,%r2)
