	.text
	.def	 foo5;
	.scl	2;
	.type	32;
	.endef
	.globl	foo5
	.align	16, 0x90
foo5:                                   # @foo5
.Ltmp0:
.seh_proc foo5
# BB#0:                                 # %entry
	pushq	%rbp
.Ltmp1:
	.seh_pushreg 5
	pushq	%rdi
.Ltmp2:
	.seh_pushreg 7
	pushq	%rbx
.Ltmp3:
	.seh_pushreg 3
	subq	$384, %rsp              # imm = 0x180
.Ltmp4:
	.seh_stackalloc 384
	leaq	128(%rsp), %rbp
.Ltmp5:
	.seh_setframe 5, 128
	movaps	%xmm7, -32(%rbp)        # 16-byte Spill
	movaps	%xmm6, -48(%rbp)        # 16-byte Spill
.Ltmp6:
	.seh_savexmm 6, 80
.Ltmp7:
	.seh_savexmm 7, 96
.Ltmp8:
	.seh_endprologue
	andq	$-64, %rsp
	#APP
	#NO_APP
	movl	$42, (%rsp)
	movaps	-48(%rbp), %xmm6        # 16-byte Reload
	movaps	-32(%rbp), %xmm7        # 16-byte Reload
	leaq	256(%rbp), %rsp
	popq	%rbx
	popq	%rdi
	popq	%rbp
	retq
.Leh_func_end0:
.Ltmp9:
	.seh_endproc


