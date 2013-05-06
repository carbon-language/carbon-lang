# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: %r0 used in an address
#CHECK: sll	%r2,8(%r0)
#CHECK: error: %r0 used in an address
#CHECK: br	%r0
#CHECK: error: %r0 used in an address
#CHECK: l	%r1,8(%r0)
#CHECK: error: %r0 used in an address
#CHECK: l	%r1,8(%r0,%r15)
#CHECK: error: %r0 used in an address
#CHECK: l	%r1,8(%r15,%r0)

	sll	%r2,8(%r0)
	br	%r0
	l	%r1,8(%r0)
	l	%r1,8(%r0,%r15)
	l	%r1,8(%r15,%r0)
