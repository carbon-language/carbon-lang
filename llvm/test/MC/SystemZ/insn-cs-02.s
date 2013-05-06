# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: cs	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cs	%r0, %r0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: cs	%r0, %r0, 0(%r1,%r2)

	cs	%r0, %r0, -1
	cs	%r0, %r0, 4096
	cs	%r0, %r0, 0(%r1,%r2)
