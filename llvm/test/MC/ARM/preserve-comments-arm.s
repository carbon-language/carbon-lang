	@RUN: llvm-mc -preserve-comments -n -triple arm-eabi < %s > %t
	@RUN: sed 's/[\t]#/	@/g' %s > %t2
	@RUN: diff %t %t2
	.text

	mov	r0, r0
foo:	#Comment here
	mov	r0, r0	@ EOL comment
	.ident	""

