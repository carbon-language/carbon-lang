// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

// CHECK: 	movb	$127, 3735928559(%ebx,%ecx,8)
        	movb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movw	$31438, 3735928559(%ebx,%ecx,8)
        	movw	$0x7ace,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movl	$2063514302, 3735928559(%ebx,%ecx,8)
        	movl	$0x7afebabe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movl	$324478056, 3735928559(%ebx,%ecx,8)
        	movl	$0x13572468,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movsbl	3735928559(%ebx,%ecx,8), %ecx
        	movsbl	0xdeadbeef(%ebx,%ecx,8),%ecx

// CHECK: 	movswl	3735928559(%ebx,%ecx,8), %ecx
        	movswl	0xdeadbeef(%ebx,%ecx,8),%ecx

// CHECK: 	movzbl	3735928559(%ebx,%ecx,8), %ecx  # NOREX
        	movzbl	0xdeadbeef(%ebx,%ecx,8),%ecx

// CHECK: 	movzwl	3735928559(%ebx,%ecx,8), %ecx
        	movzwl	0xdeadbeef(%ebx,%ecx,8),%ecx

// CHECK: 	pushl	3735928559(%ebx,%ecx,8)
        	pushl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	popl	3735928559(%ebx,%ecx,8)
        	popl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	lahf
        	lahf

// CHECK: 	sahf
        	sahf

// CHECK: 	addb	$254, 3735928559(%ebx,%ecx,8)
        	addb	$0xfe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	addb	$127, 3735928559(%ebx,%ecx,8)
        	addb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	addw	$31438, 3735928559(%ebx,%ecx,8)
        	addw	$0x7ace,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	addl	$2063514302, 3735928559(%ebx,%ecx,8)
        	addl	$0x7afebabe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	addl	$324478056, 3735928559(%ebx,%ecx,8)
        	addl	$0x13572468,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	incl	3735928559(%ebx,%ecx,8)
        	incl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	subb	$254, 3735928559(%ebx,%ecx,8)
        	subb	$0xfe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	subb	$127, 3735928559(%ebx,%ecx,8)
        	subb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	subw	$31438, 3735928559(%ebx,%ecx,8)
        	subw	$0x7ace,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	subl	$2063514302, 3735928559(%ebx,%ecx,8)
        	subl	$0x7afebabe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	subl	$324478056, 3735928559(%ebx,%ecx,8)
        	subl	$0x13572468,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	decl	3735928559(%ebx,%ecx,8)
        	decl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	sbbw	$31438, 3735928559(%ebx,%ecx,8)
        	sbbw	$0x7ace,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	sbbl	$2063514302, 3735928559(%ebx,%ecx,8)
        	sbbl	$0x7afebabe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	sbbl	$324478056, 3735928559(%ebx,%ecx,8)
        	sbbl	$0x13572468,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	cmpb	$254, 3735928559(%ebx,%ecx,8)
        	cmpb	$0xfe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	cmpb	$127, 3735928559(%ebx,%ecx,8)
        	cmpb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	cmpw	$31438, 3735928559(%ebx,%ecx,8)
        	cmpw	$0x7ace,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	cmpl	$2063514302, 3735928559(%ebx,%ecx,8)
        	cmpl	$0x7afebabe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	cmpl	$324478056, 3735928559(%ebx,%ecx,8)
        	cmpl	$0x13572468,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	testb	$127, 3735928559(%ebx,%ecx,8)
        	testb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	testw	$31438, 3735928559(%ebx,%ecx,8)
        	testw	$0x7ace,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	testl	$2063514302, 3735928559(%ebx,%ecx,8)
        	testl	$0x7afebabe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	testl	$324478056, 3735928559(%ebx,%ecx,8)
        	testl	$0x13572468,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	andb	$254, 3735928559(%ebx,%ecx,8)
        	andb	$0xfe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	andb	$127, 3735928559(%ebx,%ecx,8)
        	andb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	andw	$31438, 3735928559(%ebx,%ecx,8)
        	andw	$0x7ace,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	andl	$2063514302, 3735928559(%ebx,%ecx,8)
        	andl	$0x7afebabe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	andl	$324478056, 3735928559(%ebx,%ecx,8)
        	andl	$0x13572468,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	orb	$254, 3735928559(%ebx,%ecx,8)
        	orb	$0xfe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	orb	$127, 3735928559(%ebx,%ecx,8)
        	orb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	orw	$31438, 3735928559(%ebx,%ecx,8)
        	orw	$0x7ace,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	orl	$2063514302, 3735928559(%ebx,%ecx,8)
        	orl	$0x7afebabe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	orl	$324478056, 3735928559(%ebx,%ecx,8)
        	orl	$0x13572468,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	xorb	$254, 3735928559(%ebx,%ecx,8)
        	xorb	$0xfe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	xorb	$127, 3735928559(%ebx,%ecx,8)
        	xorb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	xorw	$31438, 3735928559(%ebx,%ecx,8)
        	xorw	$0x7ace,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	xorl	$2063514302, 3735928559(%ebx,%ecx,8)
        	xorl	$0x7afebabe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	xorl	$324478056, 3735928559(%ebx,%ecx,8)
        	xorl	$0x13572468,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	adcb	$254, 3735928559(%ebx,%ecx,8)
        	adcb	$0xfe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	adcb	$127, 3735928559(%ebx,%ecx,8)
        	adcb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	adcw	$31438, 3735928559(%ebx,%ecx,8)
        	adcw	$0x7ace,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	adcl	$2063514302, 3735928559(%ebx,%ecx,8)
        	adcl	$0x7afebabe,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	adcl	$324478056, 3735928559(%ebx,%ecx,8)
        	adcl	$0x13572468,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	negl	3735928559(%ebx,%ecx,8)
        	negl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	notl	3735928559(%ebx,%ecx,8)
        	notl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	cbtw
        	cbtw

// CHECK: 	cwtl
        	cwtl

// CHECK: 	cwtd
        	cwtd

// CHECK: 	cltd
        	cltd

// CHECK: 	mull	3735928559(%ebx,%ecx,8)
        	mull	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	imull	3735928559(%ebx,%ecx,8)
        	imull	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	divl	3735928559(%ebx,%ecx,8)
        	divl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	idivl	3735928559(%ebx,%ecx,8)
        	idivl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	roll	$0, 3735928559(%ebx,%ecx,8)
        	roll	$0,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	rolb	$127, 3735928559(%ebx,%ecx,8)
        	rolb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	roll	3735928559(%ebx,%ecx,8)
        	roll	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	rorl	$0, 3735928559(%ebx,%ecx,8)
        	rorl	$0,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	rorb	$127, 3735928559(%ebx,%ecx,8)
        	rorb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	rorl	3735928559(%ebx,%ecx,8)
        	rorl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	shll	$0, 3735928559(%ebx,%ecx,8)
        	shll	$0,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	shlb	$127, 3735928559(%ebx,%ecx,8)
        	shlb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	shll	3735928559(%ebx,%ecx,8)
        	shll	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	shrl	$0, 3735928559(%ebx,%ecx,8)
        	shrl	$0,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	shrb	$127, 3735928559(%ebx,%ecx,8)
        	shrb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	shrl	3735928559(%ebx,%ecx,8)
        	shrl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	sarl	$0, 3735928559(%ebx,%ecx,8)
        	sarl	$0,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	sarb	$127, 3735928559(%ebx,%ecx,8)
        	sarb	$0x7f,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	sarl	3735928559(%ebx,%ecx,8)
        	sarl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	call	*%ecx
        	call	*%ecx

// CHECK: 	call	*3735928559(%ebx,%ecx,8)
        	call	*0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	call	*3735928559(%ebx,%ecx,8)
        	call	*0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	jmp	*3735928559(%ebx,%ecx,8)  # TAILCALL
        	jmp	*0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	jmp	*3735928559(%ebx,%ecx,8)  # TAILCALL
        	jmp	*0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	ljmpl	*3735928559(%ebx,%ecx,8)
        	ljmpl	*0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	lret
        	lret

// CHECK: 	leave
        	leave

// CHECK: 	seto	%bl
        	seto	%bl

// CHECK: 	seto	3735928559(%ebx,%ecx,8)
        	seto	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setno	%bl
        	setno	%bl

// CHECK: 	setno	3735928559(%ebx,%ecx,8)
        	setno	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setb	%bl
        	setb	%bl

// CHECK: 	setb	3735928559(%ebx,%ecx,8)
        	setb	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setae	%bl
        	setae	%bl

// CHECK: 	setae	3735928559(%ebx,%ecx,8)
        	setae	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	sete	%bl
        	sete	%bl

// CHECK: 	sete	3735928559(%ebx,%ecx,8)
        	sete	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setne	%bl
        	setne	%bl

// CHECK: 	setne	3735928559(%ebx,%ecx,8)
        	setne	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setbe	%bl
        	setbe	%bl

// CHECK: 	setbe	3735928559(%ebx,%ecx,8)
        	setbe	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	seta	%bl
        	seta	%bl

// CHECK: 	seta	3735928559(%ebx,%ecx,8)
        	seta	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	sets	%bl
        	sets	%bl

// CHECK: 	sets	3735928559(%ebx,%ecx,8)
        	sets	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setns	%bl
        	setns	%bl

// CHECK: 	setns	3735928559(%ebx,%ecx,8)
        	setns	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setp	%bl
        	setp	%bl

// CHECK: 	setp	3735928559(%ebx,%ecx,8)
        	setp	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setnp	%bl
        	setnp	%bl

// CHECK: 	setnp	3735928559(%ebx,%ecx,8)
        	setnp	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setl	%bl
        	setl	%bl

// CHECK: 	setl	3735928559(%ebx,%ecx,8)
        	setl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setge	%bl
        	setge	%bl

// CHECK: 	setge	3735928559(%ebx,%ecx,8)
        	setge	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setle	%bl
        	setle	%bl

// CHECK: 	setle	3735928559(%ebx,%ecx,8)
        	setle	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	setg	%bl
        	setg	%bl

// CHECK: 	setg	3735928559(%ebx,%ecx,8)
        	setg	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	nopl	3735928559(%ebx,%ecx,8)
        	nopl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	nop
        	nop

// CHECK: 	fldl	3735928559(%ebx,%ecx,8)
        	fldl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fildl	3735928559(%ebx,%ecx,8)
        	fildl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fildll	3735928559(%ebx,%ecx,8)
        	fildll	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fldt	3735928559(%ebx,%ecx,8)
        	fldt	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fbld	3735928559(%ebx,%ecx,8)
        	fbld	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fstl	3735928559(%ebx,%ecx,8)
        	fstl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fistl	3735928559(%ebx,%ecx,8)
        	fistl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fstpl	3735928559(%ebx,%ecx,8)
        	fstpl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fistpl	3735928559(%ebx,%ecx,8)
        	fistpl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fistpll	3735928559(%ebx,%ecx,8)
        	fistpll	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fstpt	3735928559(%ebx,%ecx,8)
        	fstpt	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fbstp	3735928559(%ebx,%ecx,8)
        	fbstp	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	ficoml	3735928559(%ebx,%ecx,8)
        	ficoml	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	ficompl	3735928559(%ebx,%ecx,8)
        	ficompl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fucompp
        	fucompp

// CHECK: 	ftst
        	ftst

// CHECK: 	fld1
        	fld1

// CHECK: 	fldz
        	fldz

// CHECK: 	faddl	3735928559(%ebx,%ecx,8)
        	faddl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fiaddl	3735928559(%ebx,%ecx,8)
        	fiaddl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fsubl	3735928559(%ebx,%ecx,8)
        	fsubl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fisubl	3735928559(%ebx,%ecx,8)
        	fisubl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fsubrl	3735928559(%ebx,%ecx,8)
        	fsubrl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fisubrl	3735928559(%ebx,%ecx,8)
        	fisubrl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fmull	3735928559(%ebx,%ecx,8)
        	fmull	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fimull	3735928559(%ebx,%ecx,8)
        	fimull	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fdivl	3735928559(%ebx,%ecx,8)
        	fdivl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fidivl	3735928559(%ebx,%ecx,8)
        	fidivl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fdivrl	3735928559(%ebx,%ecx,8)
        	fdivrl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fidivrl	3735928559(%ebx,%ecx,8)
        	fidivrl	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fsqrt
        	fsqrt

// CHECK: 	fsin
        	fsin

// CHECK: 	fcos
        	fcos

// CHECK: 	fchs
        	fchs

// CHECK: 	fabs
        	fabs

// CHECK: 	fldcw	3735928559(%ebx,%ecx,8)
        	fldcw	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	fnstcw	3735928559(%ebx,%ecx,8)
        	fnstcw	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	rdtsc
        	rdtsc

// CHECK: 	sysenter
        	sysenter

// CHECK: 	sysexit
        	sysexit

// CHECK: 	ud2
        	ud2

// CHECK: 	movnti	%ecx, 3735928559(%ebx,%ecx,8)
        	movnti	%ecx,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	clflush	3735928559(%ebx,%ecx,8)
        	clflush	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	emms
        	emms

// CHECK: 	movd	%ecx, %mm3
        	movd	%ecx,%mm3

// CHECK: 	movd	3735928559(%ebx,%ecx,8), %mm3
        	movd	0xdeadbeef(%ebx,%ecx,8),%mm3

// CHECK: 	movd	%ecx, %xmm5
        	movd	%ecx,%xmm5

// CHECK: 	movd	3735928559(%ebx,%ecx,8), %xmm5
        	movd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movd	%xmm5, %ecx
        	movd	%xmm5,%ecx

// CHECK: 	movd	%xmm5, 3735928559(%ebx,%ecx,8)
        	movd	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movq	3735928559(%ebx,%ecx,8), %mm3
        	movq	0xdeadbeef(%ebx,%ecx,8),%mm3

// CHECK: 	movq	%mm3, %mm3
        	movq	%mm3,%mm3

// CHECK: 	movq	%mm3, %mm3
        	movq	%mm3,%mm3

// CHECK: 	movq	%xmm5, %xmm5
        	movq	%xmm5,%xmm5

// CHECK: 	movq	%xmm5, %xmm5
        	movq	%xmm5,%xmm5

// CHECK: 	packssdw	%mm3, %mm3
        	packssdw	%mm3,%mm3

// CHECK: 	packssdw	%xmm5, %xmm5
        	packssdw	%xmm5,%xmm5

// CHECK: 	packsswb	%mm3, %mm3
        	packsswb	%mm3,%mm3

// CHECK: 	packsswb	%xmm5, %xmm5
        	packsswb	%xmm5,%xmm5

// CHECK: 	packuswb	%mm3, %mm3
        	packuswb	%mm3,%mm3

// CHECK: 	packuswb	%xmm5, %xmm5
        	packuswb	%xmm5,%xmm5

// CHECK: 	paddb	%mm3, %mm3
        	paddb	%mm3,%mm3

// CHECK: 	paddb	%xmm5, %xmm5
        	paddb	%xmm5,%xmm5

// CHECK: 	paddw	%mm3, %mm3
        	paddw	%mm3,%mm3

// CHECK: 	paddw	%xmm5, %xmm5
        	paddw	%xmm5,%xmm5

// CHECK: 	paddd	%mm3, %mm3
        	paddd	%mm3,%mm3

// CHECK: 	paddd	%xmm5, %xmm5
        	paddd	%xmm5,%xmm5

// CHECK: 	paddq	%mm3, %mm3
        	paddq	%mm3,%mm3

// CHECK: 	paddq	%xmm5, %xmm5
        	paddq	%xmm5,%xmm5

// CHECK: 	paddsb	%mm3, %mm3
        	paddsb	%mm3,%mm3

// CHECK: 	paddsb	%xmm5, %xmm5
        	paddsb	%xmm5,%xmm5

// CHECK: 	paddsw	%mm3, %mm3
        	paddsw	%mm3,%mm3

// CHECK: 	paddsw	%xmm5, %xmm5
        	paddsw	%xmm5,%xmm5

// CHECK: 	paddusb	%mm3, %mm3
        	paddusb	%mm3,%mm3

// CHECK: 	paddusb	%xmm5, %xmm5
        	paddusb	%xmm5,%xmm5

// CHECK: 	paddusw	%mm3, %mm3
        	paddusw	%mm3,%mm3

// CHECK: 	paddusw	%xmm5, %xmm5
        	paddusw	%xmm5,%xmm5

// CHECK: 	pand	%mm3, %mm3
        	pand	%mm3,%mm3

// CHECK: 	pand	%xmm5, %xmm5
        	pand	%xmm5,%xmm5

// CHECK: 	pandn	%mm3, %mm3
        	pandn	%mm3,%mm3

// CHECK: 	pandn	%xmm5, %xmm5
        	pandn	%xmm5,%xmm5

// CHECK: 	pcmpeqb	%mm3, %mm3
        	pcmpeqb	%mm3,%mm3

// CHECK: 	pcmpeqb	%xmm5, %xmm5
        	pcmpeqb	%xmm5,%xmm5

// CHECK: 	pcmpeqw	%mm3, %mm3
        	pcmpeqw	%mm3,%mm3

// CHECK: 	pcmpeqw	%xmm5, %xmm5
        	pcmpeqw	%xmm5,%xmm5

// CHECK: 	pcmpeqd	%mm3, %mm3
        	pcmpeqd	%mm3,%mm3

// CHECK: 	pcmpeqd	%xmm5, %xmm5
        	pcmpeqd	%xmm5,%xmm5

// CHECK: 	pcmpgtb	%mm3, %mm3
        	pcmpgtb	%mm3,%mm3

// CHECK: 	pcmpgtb	%xmm5, %xmm5
        	pcmpgtb	%xmm5,%xmm5

// CHECK: 	pcmpgtw	%mm3, %mm3
        	pcmpgtw	%mm3,%mm3

// CHECK: 	pcmpgtw	%xmm5, %xmm5
        	pcmpgtw	%xmm5,%xmm5

// CHECK: 	pcmpgtd	%mm3, %mm3
        	pcmpgtd	%mm3,%mm3

// CHECK: 	pcmpgtd	%xmm5, %xmm5
        	pcmpgtd	%xmm5,%xmm5

// CHECK: 	pmaddwd	%mm3, %mm3
        	pmaddwd	%mm3,%mm3

// CHECK: 	pmaddwd	%xmm5, %xmm5
        	pmaddwd	%xmm5,%xmm5

// CHECK: 	pmulhw	%mm3, %mm3
        	pmulhw	%mm3,%mm3

// CHECK: 	pmulhw	%xmm5, %xmm5
        	pmulhw	%xmm5,%xmm5

// CHECK: 	pmullw	%mm3, %mm3
        	pmullw	%mm3,%mm3

// CHECK: 	pmullw	%xmm5, %xmm5
        	pmullw	%xmm5,%xmm5

// CHECK: 	por	%mm3, %mm3
        	por	%mm3,%mm3

// CHECK: 	por	%xmm5, %xmm5
        	por	%xmm5,%xmm5

// CHECK: 	psllw	%mm3, %mm3
        	psllw	%mm3,%mm3

// CHECK: 	psllw	%xmm5, %xmm5
        	psllw	%xmm5,%xmm5

// CHECK: 	psllw	$127, %mm3
        	psllw	$0x7f,%mm3

// CHECK: 	psllw	$127, %xmm5
        	psllw	$0x7f,%xmm5

// CHECK: 	pslld	%mm3, %mm3
        	pslld	%mm3,%mm3

// CHECK: 	pslld	%xmm5, %xmm5
        	pslld	%xmm5,%xmm5

// CHECK: 	pslld	$127, %mm3
        	pslld	$0x7f,%mm3

// CHECK: 	pslld	$127, %xmm5
        	pslld	$0x7f,%xmm5

// CHECK: 	psllq	%mm3, %mm3
        	psllq	%mm3,%mm3

// CHECK: 	psllq	%xmm5, %xmm5
        	psllq	%xmm5,%xmm5

// CHECK: 	psllq	$127, %mm3
        	psllq	$0x7f,%mm3

// CHECK: 	psllq	$127, %xmm5
        	psllq	$0x7f,%xmm5

// CHECK: 	psraw	%mm3, %mm3
        	psraw	%mm3,%mm3

// CHECK: 	psraw	%xmm5, %xmm5
        	psraw	%xmm5,%xmm5

// CHECK: 	psraw	$127, %mm3
        	psraw	$0x7f,%mm3

// CHECK: 	psraw	$127, %xmm5
        	psraw	$0x7f,%xmm5

// CHECK: 	psrad	%mm3, %mm3
        	psrad	%mm3,%mm3

// CHECK: 	psrad	%xmm5, %xmm5
        	psrad	%xmm5,%xmm5

// CHECK: 	psrad	$127, %mm3
        	psrad	$0x7f,%mm3

// CHECK: 	psrad	$127, %xmm5
        	psrad	$0x7f,%xmm5

// CHECK: 	psrlw	%mm3, %mm3
        	psrlw	%mm3,%mm3

// CHECK: 	psrlw	%xmm5, %xmm5
        	psrlw	%xmm5,%xmm5

// CHECK: 	psrlw	$127, %mm3
        	psrlw	$0x7f,%mm3

// CHECK: 	psrlw	$127, %xmm5
        	psrlw	$0x7f,%xmm5

// CHECK: 	psrld	%mm3, %mm3
        	psrld	%mm3,%mm3

// CHECK: 	psrld	%xmm5, %xmm5
        	psrld	%xmm5,%xmm5

// CHECK: 	psrld	$127, %mm3
        	psrld	$0x7f,%mm3

// CHECK: 	psrld	$127, %xmm5
        	psrld	$0x7f,%xmm5

// CHECK: 	psrlq	%mm3, %mm3
        	psrlq	%mm3,%mm3

// CHECK: 	psrlq	%xmm5, %xmm5
        	psrlq	%xmm5,%xmm5

// CHECK: 	psrlq	$127, %mm3
        	psrlq	$0x7f,%mm3

// CHECK: 	psrlq	$127, %xmm5
        	psrlq	$0x7f,%xmm5

// CHECK: 	psubb	%mm3, %mm3
        	psubb	%mm3,%mm3

// CHECK: 	psubb	%xmm5, %xmm5
        	psubb	%xmm5,%xmm5

// CHECK: 	psubw	%mm3, %mm3
        	psubw	%mm3,%mm3

// CHECK: 	psubw	%xmm5, %xmm5
        	psubw	%xmm5,%xmm5

// CHECK: 	psubd	%mm3, %mm3
        	psubd	%mm3,%mm3

// CHECK: 	psubd	%xmm5, %xmm5
        	psubd	%xmm5,%xmm5

// CHECK: 	psubq	%mm3, %mm3
        	psubq	%mm3,%mm3

// CHECK: 	psubq	%xmm5, %xmm5
        	psubq	%xmm5,%xmm5

// CHECK: 	psubsb	%mm3, %mm3
        	psubsb	%mm3,%mm3

// CHECK: 	psubsb	%xmm5, %xmm5
        	psubsb	%xmm5,%xmm5

// CHECK: 	psubsw	%mm3, %mm3
        	psubsw	%mm3,%mm3

// CHECK: 	psubsw	%xmm5, %xmm5
        	psubsw	%xmm5,%xmm5

// CHECK: 	psubusb	%mm3, %mm3
        	psubusb	%mm3,%mm3

// CHECK: 	psubusb	%xmm5, %xmm5
        	psubusb	%xmm5,%xmm5

// CHECK: 	psubusw	%mm3, %mm3
        	psubusw	%mm3,%mm3

// CHECK: 	psubusw	%xmm5, %xmm5
        	psubusw	%xmm5,%xmm5

// CHECK: 	punpckhbw	%mm3, %mm3
        	punpckhbw	%mm3,%mm3

// CHECK: 	punpckhbw	%xmm5, %xmm5
        	punpckhbw	%xmm5,%xmm5

// CHECK: 	punpckhwd	%mm3, %mm3
        	punpckhwd	%mm3,%mm3

// CHECK: 	punpckhwd	%xmm5, %xmm5
        	punpckhwd	%xmm5,%xmm5

// CHECK: 	punpckhdq	%mm3, %mm3
        	punpckhdq	%mm3,%mm3

// CHECK: 	punpckhdq	%xmm5, %xmm5
        	punpckhdq	%xmm5,%xmm5

// CHECK: 	punpcklbw	%mm3, %mm3
        	punpcklbw	%mm3,%mm3

// CHECK: 	punpcklbw	%xmm5, %xmm5
        	punpcklbw	%xmm5,%xmm5

// CHECK: 	punpcklwd	%mm3, %mm3
        	punpcklwd	%mm3,%mm3

// CHECK: 	punpcklwd	%xmm5, %xmm5
        	punpcklwd	%xmm5,%xmm5

// CHECK: 	punpckldq	%mm3, %mm3
        	punpckldq	%mm3,%mm3

// CHECK: 	punpckldq	%xmm5, %xmm5
        	punpckldq	%xmm5,%xmm5

// CHECK: 	pxor	%mm3, %mm3
        	pxor	%mm3,%mm3

// CHECK: 	pxor	%xmm5, %xmm5
        	pxor	%xmm5,%xmm5

// CHECK: 	addps	%xmm5, %xmm5
        	addps	%xmm5,%xmm5

// CHECK: 	addss	%xmm5, %xmm5
        	addss	%xmm5,%xmm5

// CHECK: 	andnps	%xmm5, %xmm5
        	andnps	%xmm5,%xmm5

// CHECK: 	andps	%xmm5, %xmm5
        	andps	%xmm5,%xmm5

// CHECK: 	cvtpi2ps	3735928559(%ebx,%ecx,8), %xmm5
        	cvtpi2ps	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	cvtpi2ps	%mm3, %xmm5
        	cvtpi2ps	%mm3,%xmm5

// CHECK: 	cvtps2pi	3735928559(%ebx,%ecx,8), %mm3
        	cvtps2pi	0xdeadbeef(%ebx,%ecx,8),%mm3

// CHECK: 	cvtps2pi	%xmm5, %mm3
        	cvtps2pi	%xmm5,%mm3

// CHECK: 	cvtsi2ss	%ecx, %xmm5
        	cvtsi2ss	%ecx,%xmm5

// CHECK: 	cvtsi2ss	3735928559(%ebx,%ecx,8), %xmm5
        	cvtsi2ss	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	cvttps2pi	3735928559(%ebx,%ecx,8), %mm3
        	cvttps2pi	0xdeadbeef(%ebx,%ecx,8),%mm3

// CHECK: 	cvttps2pi	%xmm5, %mm3
        	cvttps2pi	%xmm5,%mm3

// CHECK: 	cvttss2si	3735928559(%ebx,%ecx,8), %ecx
        	cvttss2si	0xdeadbeef(%ebx,%ecx,8),%ecx

// CHECK: 	cvttss2si	%xmm5, %ecx
        	cvttss2si	%xmm5,%ecx

// CHECK: 	divps	%xmm5, %xmm5
        	divps	%xmm5,%xmm5

// CHECK: 	divss	%xmm5, %xmm5
        	divss	%xmm5,%xmm5

// CHECK: 	ldmxcsr	3735928559(%ebx,%ecx,8)
        	ldmxcsr	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	maskmovq	%mm3, %mm3
        	maskmovq	%mm3,%mm3

// CHECK: 	maxps	%xmm5, %xmm5
        	maxps	%xmm5,%xmm5

// CHECK: 	maxss	%xmm5, %xmm5
        	maxss	%xmm5,%xmm5

// CHECK: 	minps	%xmm5, %xmm5
        	minps	%xmm5,%xmm5

// CHECK: 	minss	%xmm5, %xmm5
        	minss	%xmm5,%xmm5

// CHECK: 	movaps	3735928559(%ebx,%ecx,8), %xmm5
        	movaps	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movaps	%xmm5, %xmm5
        	movaps	%xmm5,%xmm5

// CHECK: 	movaps	%xmm5, 3735928559(%ebx,%ecx,8)
        	movaps	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movaps	%xmm5, %xmm5
        	movaps	%xmm5,%xmm5

// CHECK: 	movhlps	%xmm5, %xmm5
        	movhlps	%xmm5,%xmm5

// CHECK: 	movhps	%xmm5, 3735928559(%ebx,%ecx,8)
        	movhps	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movlhps	%xmm5, %xmm5
        	movlhps	%xmm5,%xmm5

// CHECK: 	movlps	%xmm5, 3735928559(%ebx,%ecx,8)
        	movlps	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movmskps	%xmm5, %ecx
        	movmskps	%xmm5,%ecx

// CHECK: 	movntps	%xmm5, 3735928559(%ebx,%ecx,8)
        	movntps	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movntq	%mm3, 3735928559(%ebx,%ecx,8)
        	movntq	%mm3,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movntdq	%xmm5, 3735928559(%ebx,%ecx,8)
        	movntdq	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movss	3735928559(%ebx,%ecx,8), %xmm5
        	movss	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movss	%xmm5, %xmm5
        	movss	%xmm5,%xmm5

// CHECK: 	movss	%xmm5, 3735928559(%ebx,%ecx,8)
        	movss	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movss	%xmm5, %xmm5
        	movss	%xmm5,%xmm5

// CHECK: 	movups	3735928559(%ebx,%ecx,8), %xmm5
        	movups	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movups	%xmm5, %xmm5
        	movups	%xmm5,%xmm5

// CHECK: 	movups	%xmm5, 3735928559(%ebx,%ecx,8)
        	movups	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movups	%xmm5, %xmm5
        	movups	%xmm5,%xmm5

// CHECK: 	mulps	%xmm5, %xmm5
        	mulps	%xmm5,%xmm5

// CHECK: 	mulss	%xmm5, %xmm5
        	mulss	%xmm5,%xmm5

// CHECK: 	orps	%xmm5, %xmm5
        	orps	%xmm5,%xmm5

// CHECK: 	pavgb	%mm3, %mm3
        	pavgb	%mm3,%mm3

// CHECK: 	pavgb	%xmm5, %xmm5
        	pavgb	%xmm5,%xmm5

// CHECK: 	pavgw	%mm3, %mm3
        	pavgw	%mm3,%mm3

// CHECK: 	pavgw	%xmm5, %xmm5
        	pavgw	%xmm5,%xmm5

// CHECK: 	pmaxsw	%mm3, %mm3
        	pmaxsw	%mm3,%mm3

// CHECK: 	pmaxsw	%xmm5, %xmm5
        	pmaxsw	%xmm5,%xmm5

// CHECK: 	pmaxub	%mm3, %mm3
        	pmaxub	%mm3,%mm3

// CHECK: 	pmaxub	%xmm5, %xmm5
        	pmaxub	%xmm5,%xmm5

// CHECK: 	pminsw	%mm3, %mm3
        	pminsw	%mm3,%mm3

// CHECK: 	pminsw	%xmm5, %xmm5
        	pminsw	%xmm5,%xmm5

// CHECK: 	pminub	%mm3, %mm3
        	pminub	%mm3,%mm3

// CHECK: 	pminub	%xmm5, %xmm5
        	pminub	%xmm5,%xmm5

// CHECK: 	pmovmskb	%mm3, %ecx
        	pmovmskb	%mm3,%ecx

// CHECK: 	pmovmskb	%xmm5, %ecx
        	pmovmskb	%xmm5,%ecx

// CHECK: 	pmulhuw	%mm3, %mm3
        	pmulhuw	%mm3,%mm3

// CHECK: 	pmulhuw	%xmm5, %xmm5
        	pmulhuw	%xmm5,%xmm5

// CHECK: 	prefetchnta	3735928559(%ebx,%ecx,8)
        	prefetchnta	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	prefetcht0	3735928559(%ebx,%ecx,8)
        	prefetcht0	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	prefetcht1	3735928559(%ebx,%ecx,8)
        	prefetcht1	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	prefetcht2	3735928559(%ebx,%ecx,8)
        	prefetcht2	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	psadbw	%mm3, %mm3
        	psadbw	%mm3,%mm3

// CHECK: 	psadbw	%xmm5, %xmm5
        	psadbw	%xmm5,%xmm5

// CHECK: 	rcpps	3735928559(%ebx,%ecx,8), %xmm5
        	rcpps	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	rcpps	%xmm5, %xmm5
        	rcpps	%xmm5,%xmm5

// CHECK: 	rcpss	3735928559(%ebx,%ecx,8), %xmm5
        	rcpss	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	rcpss	%xmm5, %xmm5
        	rcpss	%xmm5,%xmm5

// CHECK: 	rsqrtps	3735928559(%ebx,%ecx,8), %xmm5
        	rsqrtps	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	rsqrtps	%xmm5, %xmm5
        	rsqrtps	%xmm5,%xmm5

// CHECK: 	rsqrtss	3735928559(%ebx,%ecx,8), %xmm5
        	rsqrtss	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	rsqrtss	%xmm5, %xmm5
        	rsqrtss	%xmm5,%xmm5

// CHECK: 	sqrtps	3735928559(%ebx,%ecx,8), %xmm5
        	sqrtps	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	sqrtps	%xmm5, %xmm5
        	sqrtps	%xmm5,%xmm5

// CHECK: 	sqrtss	3735928559(%ebx,%ecx,8), %xmm5
        	sqrtss	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	sqrtss	%xmm5, %xmm5
        	sqrtss	%xmm5,%xmm5

// CHECK: 	stmxcsr	3735928559(%ebx,%ecx,8)
        	stmxcsr	0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	subps	%xmm5, %xmm5
        	subps	%xmm5,%xmm5

// CHECK: 	subss	%xmm5, %xmm5
        	subss	%xmm5,%xmm5

// CHECK: 	ucomiss	3735928559(%ebx,%ecx,8), %xmm5
        	ucomiss	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	ucomiss	%xmm5, %xmm5
        	ucomiss	%xmm5,%xmm5

// CHECK: 	unpckhps	%xmm5, %xmm5
        	unpckhps	%xmm5,%xmm5

// CHECK: 	unpcklps	%xmm5, %xmm5
        	unpcklps	%xmm5,%xmm5

// CHECK: 	xorps	%xmm5, %xmm5
        	xorps	%xmm5,%xmm5

// CHECK: 	addpd	%xmm5, %xmm5
        	addpd	%xmm5,%xmm5

// CHECK: 	addsd	%xmm5, %xmm5
        	addsd	%xmm5,%xmm5

// CHECK: 	andnpd	%xmm5, %xmm5
        	andnpd	%xmm5,%xmm5

// CHECK: 	andpd	%xmm5, %xmm5
        	andpd	%xmm5,%xmm5

// CHECK: 	comisd	3735928559(%ebx,%ecx,8), %xmm5
        	comisd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	comisd	%xmm5, %xmm5
        	comisd	%xmm5,%xmm5

// CHECK: 	cvtpi2pd	3735928559(%ebx,%ecx,8), %xmm5
        	cvtpi2pd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	cvtpi2pd	%mm3, %xmm5
        	cvtpi2pd	%mm3,%xmm5

// CHECK: 	cvtsi2sd	%ecx, %xmm5
        	cvtsi2sd	%ecx,%xmm5

// CHECK: 	cvtsi2sd	3735928559(%ebx,%ecx,8), %xmm5
        	cvtsi2sd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	divpd	%xmm5, %xmm5
        	divpd	%xmm5,%xmm5

// CHECK: 	divsd	%xmm5, %xmm5
        	divsd	%xmm5,%xmm5

// CHECK: 	maxpd	%xmm5, %xmm5
        	maxpd	%xmm5,%xmm5

// CHECK: 	maxsd	%xmm5, %xmm5
        	maxsd	%xmm5,%xmm5

// CHECK: 	minpd	%xmm5, %xmm5
        	minpd	%xmm5,%xmm5

// CHECK: 	minsd	%xmm5, %xmm5
        	minsd	%xmm5,%xmm5

// CHECK: 	movapd	3735928559(%ebx,%ecx,8), %xmm5
        	movapd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movapd	%xmm5, %xmm5
        	movapd	%xmm5,%xmm5

// CHECK: 	movapd	%xmm5, 3735928559(%ebx,%ecx,8)
        	movapd	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movapd	%xmm5, %xmm5
        	movapd	%xmm5,%xmm5

// CHECK: 	movhpd	%xmm5, 3735928559(%ebx,%ecx,8)
        	movhpd	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movlpd	%xmm5, 3735928559(%ebx,%ecx,8)
        	movlpd	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movmskpd	%xmm5, %ecx
        	movmskpd	%xmm5,%ecx

// CHECK: 	movntpd	%xmm5, 3735928559(%ebx,%ecx,8)
        	movntpd	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movsd	3735928559(%ebx,%ecx,8), %xmm5
        	movsd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movsd	%xmm5, %xmm5
        	movsd	%xmm5,%xmm5

// CHECK: 	movsd	%xmm5, 3735928559(%ebx,%ecx,8)
        	movsd	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movsd	%xmm5, %xmm5
        	movsd	%xmm5,%xmm5

// CHECK: 	movupd	3735928559(%ebx,%ecx,8), %xmm5
        	movupd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movupd	%xmm5, %xmm5
        	movupd	%xmm5,%xmm5

// CHECK: 	movupd	%xmm5, 3735928559(%ebx,%ecx,8)
        	movupd	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movupd	%xmm5, %xmm5
        	movupd	%xmm5,%xmm5

// CHECK: 	mulpd	%xmm5, %xmm5
        	mulpd	%xmm5,%xmm5

// CHECK: 	mulsd	%xmm5, %xmm5
        	mulsd	%xmm5,%xmm5

// CHECK: 	orpd	%xmm5, %xmm5
        	orpd	%xmm5,%xmm5

// CHECK: 	sqrtpd	3735928559(%ebx,%ecx,8), %xmm5
        	sqrtpd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	sqrtpd	%xmm5, %xmm5
        	sqrtpd	%xmm5,%xmm5

// CHECK: 	sqrtsd	3735928559(%ebx,%ecx,8), %xmm5
        	sqrtsd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	sqrtsd	%xmm5, %xmm5
        	sqrtsd	%xmm5,%xmm5

// CHECK: 	subpd	%xmm5, %xmm5
        	subpd	%xmm5,%xmm5

// CHECK: 	subsd	%xmm5, %xmm5
        	subsd	%xmm5,%xmm5

// CHECK: 	ucomisd	3735928559(%ebx,%ecx,8), %xmm5
        	ucomisd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	ucomisd	%xmm5, %xmm5
        	ucomisd	%xmm5,%xmm5

// CHECK: 	unpckhpd	%xmm5, %xmm5
        	unpckhpd	%xmm5,%xmm5

// CHECK: 	unpcklpd	%xmm5, %xmm5
        	unpcklpd	%xmm5,%xmm5

// CHECK: 	xorpd	%xmm5, %xmm5
        	xorpd	%xmm5,%xmm5

// CHECK: 	cvtdq2pd	3735928559(%ebx,%ecx,8), %xmm5
        	cvtdq2pd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	cvtdq2pd	%xmm5, %xmm5
        	cvtdq2pd	%xmm5,%xmm5

// CHECK: 	cvtpd2dq	3735928559(%ebx,%ecx,8), %xmm5
        	cvtpd2dq	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	cvtpd2dq	%xmm5, %xmm5
        	cvtpd2dq	%xmm5,%xmm5

// CHECK: 	cvtdq2ps	3735928559(%ebx,%ecx,8), %xmm5
        	cvtdq2ps	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	cvtdq2ps	%xmm5, %xmm5
        	cvtdq2ps	%xmm5,%xmm5

// CHECK: 	cvtpd2pi	3735928559(%ebx,%ecx,8), %mm3
        	cvtpd2pi	0xdeadbeef(%ebx,%ecx,8),%mm3

// CHECK: 	cvtpd2pi	%xmm5, %mm3
        	cvtpd2pi	%xmm5,%mm3

// CHECK: 	cvtps2dq	3735928559(%ebx,%ecx,8), %xmm5
        	cvtps2dq	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	cvtps2dq	%xmm5, %xmm5
        	cvtps2dq	%xmm5,%xmm5

// CHECK: 	cvtsd2ss	3735928559(%ebx,%ecx,8), %xmm5
        	cvtsd2ss	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	cvtsd2ss	%xmm5, %xmm5
        	cvtsd2ss	%xmm5,%xmm5

// CHECK: 	cvtss2sd	3735928559(%ebx,%ecx,8), %xmm5
        	cvtss2sd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	cvtss2sd	%xmm5, %xmm5
        	cvtss2sd	%xmm5,%xmm5

// CHECK: 	cvttpd2pi	3735928559(%ebx,%ecx,8), %mm3
        	cvttpd2pi	0xdeadbeef(%ebx,%ecx,8),%mm3

// CHECK: 	cvttpd2pi	%xmm5, %mm3
        	cvttpd2pi	%xmm5,%mm3

// CHECK: 	cvttsd2si	3735928559(%ebx,%ecx,8), %ecx
        	cvttsd2si	0xdeadbeef(%ebx,%ecx,8),%ecx

// CHECK: 	cvttsd2si	%xmm5, %ecx
        	cvttsd2si	%xmm5,%ecx

// CHECK: 	maskmovdqu	%xmm5, %xmm5
        	maskmovdqu	%xmm5,%xmm5

// CHECK: 	movdqa	3735928559(%ebx,%ecx,8), %xmm5
        	movdqa	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movdqa	%xmm5, %xmm5
        	movdqa	%xmm5,%xmm5

// CHECK: 	movdqa	%xmm5, 3735928559(%ebx,%ecx,8)
        	movdqa	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movdqa	%xmm5, %xmm5
        	movdqa	%xmm5,%xmm5

// CHECK: 	movdqu	3735928559(%ebx,%ecx,8), %xmm5
        	movdqu	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movdqu	%xmm5, 3735928559(%ebx,%ecx,8)
        	movdqu	%xmm5,0xdeadbeef(%ebx,%ecx,8)

// CHECK: 	movdq2q	%xmm5, %mm3
        	movdq2q	%xmm5,%mm3

// CHECK: 	movq2dq	%mm3, %xmm5
        	movq2dq	%mm3,%xmm5

// CHECK: 	pmuludq	%mm3, %mm3
        	pmuludq	%mm3,%mm3

// CHECK: 	pmuludq	%xmm5, %xmm5
        	pmuludq	%xmm5,%xmm5

// CHECK: 	pslldq	$127, %xmm5
        	pslldq	$0x7f,%xmm5

// CHECK: 	psrldq	$127, %xmm5
        	psrldq	$0x7f,%xmm5

// CHECK: 	punpckhqdq	%xmm5, %xmm5
        	punpckhqdq	%xmm5,%xmm5

// CHECK: 	punpcklqdq	%xmm5, %xmm5
        	punpcklqdq	%xmm5,%xmm5

// CHECK: 	addsubpd	%xmm5, %xmm5
        	addsubpd	%xmm5,%xmm5

// CHECK: 	addsubps	%xmm5, %xmm5
        	addsubps	%xmm5,%xmm5

// CHECK: 	haddpd	%xmm5, %xmm5
        	haddpd	%xmm5,%xmm5

// CHECK: 	haddps	%xmm5, %xmm5
        	haddps	%xmm5,%xmm5

// CHECK: 	hsubpd	%xmm5, %xmm5
        	hsubpd	%xmm5,%xmm5

// CHECK: 	hsubps	%xmm5, %xmm5
        	hsubps	%xmm5,%xmm5

// CHECK: 	lddqu	3735928559(%ebx,%ecx,8), %xmm5
        	lddqu	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movddup	3735928559(%ebx,%ecx,8), %xmm5
        	movddup	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movddup	%xmm5, %xmm5
        	movddup	%xmm5,%xmm5

// CHECK: 	movshdup	3735928559(%ebx,%ecx,8), %xmm5
        	movshdup	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movshdup	%xmm5, %xmm5
        	movshdup	%xmm5,%xmm5

// CHECK: 	movsldup	3735928559(%ebx,%ecx,8), %xmm5
        	movsldup	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	movsldup	%xmm5, %xmm5
        	movsldup	%xmm5,%xmm5

// CHECK: 	phaddw	%mm3, %mm3
        	phaddw	%mm3,%mm3

// CHECK: 	phaddw	%xmm5, %xmm5
        	phaddw	%xmm5,%xmm5

// CHECK: 	phaddd	%mm3, %mm3
        	phaddd	%mm3,%mm3

// CHECK: 	phaddd	%xmm5, %xmm5
        	phaddd	%xmm5,%xmm5

// CHECK: 	phaddsw	%mm3, %mm3
        	phaddsw	%mm3,%mm3

// CHECK: 	phaddsw	%xmm5, %xmm5
        	phaddsw	%xmm5,%xmm5

// CHECK: 	phsubw	%mm3, %mm3
        	phsubw	%mm3,%mm3

// CHECK: 	phsubw	%xmm5, %xmm5
        	phsubw	%xmm5,%xmm5

// CHECK: 	phsubd	%mm3, %mm3
        	phsubd	%mm3,%mm3

// CHECK: 	phsubd	%xmm5, %xmm5
        	phsubd	%xmm5,%xmm5

// CHECK: 	phsubsw	%mm3, %mm3
        	phsubsw	%mm3,%mm3

// CHECK: 	phsubsw	%xmm5, %xmm5
        	phsubsw	%xmm5,%xmm5

// CHECK: 	pmaddubsw	%mm3, %mm3
        	pmaddubsw	%mm3,%mm3

// CHECK: 	pmaddubsw	%xmm5, %xmm5
        	pmaddubsw	%xmm5,%xmm5

// CHECK: 	pmulhrsw	%mm3, %mm3
        	pmulhrsw	%mm3,%mm3

// CHECK: 	pmulhrsw	%xmm5, %xmm5
        	pmulhrsw	%xmm5,%xmm5

// CHECK: 	pshufb	%mm3, %mm3
        	pshufb	%mm3,%mm3

// CHECK: 	pshufb	%xmm5, %xmm5
        	pshufb	%xmm5,%xmm5

// CHECK: 	psignb	%mm3, %mm3
        	psignb	%mm3,%mm3

// CHECK: 	psignb	%xmm5, %xmm5
        	psignb	%xmm5,%xmm5

// CHECK: 	psignw	%mm3, %mm3
        	psignw	%mm3,%mm3

// CHECK: 	psignw	%xmm5, %xmm5
        	psignw	%xmm5,%xmm5

// CHECK: 	psignd	%mm3, %mm3
        	psignd	%mm3,%mm3

// CHECK: 	psignd	%xmm5, %xmm5
        	psignd	%xmm5,%xmm5

// CHECK: 	pabsb	3735928559(%ebx,%ecx,8), %mm3
        	pabsb	0xdeadbeef(%ebx,%ecx,8),%mm3

// CHECK: 	pabsb	%mm3, %mm3
        	pabsb	%mm3,%mm3

// CHECK: 	pabsb	3735928559(%ebx,%ecx,8), %xmm5
        	pabsb	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pabsb	%xmm5, %xmm5
        	pabsb	%xmm5,%xmm5

// CHECK: 	pabsw	3735928559(%ebx,%ecx,8), %mm3
        	pabsw	0xdeadbeef(%ebx,%ecx,8),%mm3

// CHECK: 	pabsw	%mm3, %mm3
        	pabsw	%mm3,%mm3

// CHECK: 	pabsw	3735928559(%ebx,%ecx,8), %xmm5
        	pabsw	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pabsw	%xmm5, %xmm5
        	pabsw	%xmm5,%xmm5

// CHECK: 	pabsd	3735928559(%ebx,%ecx,8), %mm3
        	pabsd	0xdeadbeef(%ebx,%ecx,8),%mm3

// CHECK: 	pabsd	%mm3, %mm3
        	pabsd	%mm3,%mm3

// CHECK: 	pabsd	3735928559(%ebx,%ecx,8), %xmm5
        	pabsd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pabsd	%xmm5, %xmm5
        	pabsd	%xmm5,%xmm5

// CHECK: 	femms
        	femms

// CHECK: 	packusdw	%xmm5, %xmm5
        	packusdw	%xmm5,%xmm5

// CHECK: 	pcmpeqq	%xmm5, %xmm5
        	pcmpeqq	%xmm5,%xmm5

// CHECK: 	phminposuw	3735928559(%ebx,%ecx,8), %xmm5
        	phminposuw	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	phminposuw	%xmm5, %xmm5
        	phminposuw	%xmm5,%xmm5

// CHECK: 	pmaxsb	%xmm5, %xmm5
        	pmaxsb	%xmm5,%xmm5

// CHECK: 	pmaxsd	%xmm5, %xmm5
        	pmaxsd	%xmm5,%xmm5

// CHECK: 	pmaxud	%xmm5, %xmm5
        	pmaxud	%xmm5,%xmm5

// CHECK: 	pmaxuw	%xmm5, %xmm5
        	pmaxuw	%xmm5,%xmm5

// CHECK: 	pminsb	%xmm5, %xmm5
        	pminsb	%xmm5,%xmm5

// CHECK: 	pminsd	%xmm5, %xmm5
        	pminsd	%xmm5,%xmm5

// CHECK: 	pminud	%xmm5, %xmm5
        	pminud	%xmm5,%xmm5

// CHECK: 	pminuw	%xmm5, %xmm5
        	pminuw	%xmm5,%xmm5

// CHECK: 	pmovsxbw	3735928559(%ebx,%ecx,8), %xmm5
        	pmovsxbw	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovsxbw	%xmm5, %xmm5
        	pmovsxbw	%xmm5,%xmm5

// CHECK: 	pmovsxbd	3735928559(%ebx,%ecx,8), %xmm5
        	pmovsxbd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovsxbd	%xmm5, %xmm5
        	pmovsxbd	%xmm5,%xmm5

// CHECK: 	pmovsxbq	3735928559(%ebx,%ecx,8), %xmm5
        	pmovsxbq	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovsxbq	%xmm5, %xmm5
        	pmovsxbq	%xmm5,%xmm5

// CHECK: 	pmovsxwd	3735928559(%ebx,%ecx,8), %xmm5
        	pmovsxwd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovsxwd	%xmm5, %xmm5
        	pmovsxwd	%xmm5,%xmm5

// CHECK: 	pmovsxwq	3735928559(%ebx,%ecx,8), %xmm5
        	pmovsxwq	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovsxwq	%xmm5, %xmm5
        	pmovsxwq	%xmm5,%xmm5

// CHECK: 	pmovsxdq	3735928559(%ebx,%ecx,8), %xmm5
        	pmovsxdq	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovsxdq	%xmm5, %xmm5
        	pmovsxdq	%xmm5,%xmm5

// CHECK: 	pmovzxbw	3735928559(%ebx,%ecx,8), %xmm5
        	pmovzxbw	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovzxbw	%xmm5, %xmm5
        	pmovzxbw	%xmm5,%xmm5

// CHECK: 	pmovzxbd	3735928559(%ebx,%ecx,8), %xmm5
        	pmovzxbd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovzxbd	%xmm5, %xmm5
        	pmovzxbd	%xmm5,%xmm5

// CHECK: 	pmovzxbq	3735928559(%ebx,%ecx,8), %xmm5
        	pmovzxbq	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovzxbq	%xmm5, %xmm5
        	pmovzxbq	%xmm5,%xmm5

// CHECK: 	pmovzxwd	3735928559(%ebx,%ecx,8), %xmm5
        	pmovzxwd	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovzxwd	%xmm5, %xmm5
        	pmovzxwd	%xmm5,%xmm5

// CHECK: 	pmovzxwq	3735928559(%ebx,%ecx,8), %xmm5
        	pmovzxwq	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovzxwq	%xmm5, %xmm5
        	pmovzxwq	%xmm5,%xmm5

// CHECK: 	pmovzxdq	3735928559(%ebx,%ecx,8), %xmm5
        	pmovzxdq	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	pmovzxdq	%xmm5, %xmm5
        	pmovzxdq	%xmm5,%xmm5

// CHECK: 	pmuldq	%xmm5, %xmm5
        	pmuldq	%xmm5,%xmm5

// CHECK: 	pmulld	%xmm5, %xmm5
        	pmulld	%xmm5,%xmm5

// CHECK: 	ptest 	3735928559(%ebx,%ecx,8), %xmm5
        	ptest	0xdeadbeef(%ebx,%ecx,8),%xmm5

// CHECK: 	ptest 	%xmm5, %xmm5
        	ptest	%xmm5,%xmm5

// CHECK: 	pcmpgtq	%xmm5, %xmm5
        	pcmpgtq	%xmm5,%xmm5
