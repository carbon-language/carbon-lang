	.text
	.def	 unaligned;
	.scl	2;
	.type	32;
	.endef
	.globl	unaligned
	.align	16, 0x90
unaligned:                              # @unaligned
# BB#0:                                 # %entry
	pushq	%rbp
	movabsq	$4096, %rax             # imm = 0x1000
	callq	__chkstk
	subq	%rax, %rsp
	leaq	128(%rsp), %rbp
	leaq	15(%rcx), %rax
	andq	$-16, %rax
	callq	__chkstk
	subq	%rax, %rsp
	movq	%rsp, %rax
	subq	$48, %rsp
	movq	%rax, 32(%rsp)
	leaq	-128(%rbp), %r9
	movq	%rcx, %r8
	callq	bar
	leaq	4016(%rbp), %rsp
	popq	%rbp
	retq


