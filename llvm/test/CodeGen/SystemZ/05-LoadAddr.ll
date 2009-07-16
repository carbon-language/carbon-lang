

	.text
	.align	4
	.globl	foo
	.type	foo,@function
foo:
.BB1_0:	# entry
	sllg	%r1, %r3, 3
	lay	%r2, 8(%r1,%r2)
	br	%r14
	.size	foo, .-foo
