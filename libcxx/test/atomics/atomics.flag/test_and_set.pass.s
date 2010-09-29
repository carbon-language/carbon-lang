	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.align	4, 0x90
_main:                                  ## @main
Leh_func_begin0:
## BB#0:
	pushl	%ebp
Ltmp0:
	movl	%esp, %ebp
Ltmp1:
	pushl	%edi
	pushl	%esi
	subl	$48, %esp
Ltmp2:
	calll	L0$pb
L0$pb:
	popl	%eax
	leal	-16(%ebp), %ecx
	movl	$5, %edx
	movl	$0, -12(%ebp)
	movl	%ecx, -20(%ebp)         ## 4-byte Spill
	movb	$0, (%ecx)
	movl	-20(%ebp), %ecx         ## 4-byte Reload
	movl	%ecx, (%esp)
	movl	$5, 4(%esp)
	movl	%eax, -24(%ebp)         ## 4-byte Spill
	movl	%edx, -28(%ebp)         ## 4-byte Spill
	calll	__ZNSt3__111atomic_flag12test_and_setENS_12memory_orderE
	andb	$1, %al
	andb	$1, %al
	movzbl	%al, %ecx
	cmpl	$0, %ecx
	sete	%al
	xorb	$1, %al
	testb	%al, %al
	jne	LBB0_1
	jmp	LBB0_2
LBB0_1:
	movl	-24(%ebp), %eax         ## 4-byte Reload
	leal	L___func__.main-L0$pb(%eax), %ecx
	leal	L_.str-L0$pb(%eax), %edx
	movl	$23, %esi
	leal	L_.str1-L0$pb(%eax), %edi
	movl	%ecx, (%esp)
	movl	%edx, 4(%esp)
	movl	$23, 8(%esp)
	movl	%edi, 12(%esp)
	movl	%esi, -32(%ebp)         ## 4-byte Spill
	calll	___assert_rtn
LBB0_2:
## BB#3:
	leal	-16(%ebp), %eax
	movl	%esp, %ecx
	movl	%eax, (%ecx)
	movl	$5, 4(%ecx)
	calll	__ZNSt3__111atomic_flag12test_and_setENS_12memory_orderE
	andb	$1, %al
	movzbl	%al, %ecx
	cmpl	$1, %ecx
	sete	%al
	xorb	$1, %al
	testb	%al, %al
	jne	LBB0_4
	jmp	LBB0_5
LBB0_4:
	movl	-24(%ebp), %eax         ## 4-byte Reload
	leal	L___func__.main-L0$pb(%eax), %ecx
	leal	L_.str-L0$pb(%eax), %edx
	movl	$24, %esi
	leal	L_.str2-L0$pb(%eax), %edi
	movl	%ecx, (%esp)
	movl	%edx, 4(%esp)
	movl	$24, 8(%esp)
	movl	%edi, 12(%esp)
	movl	%esi, -36(%ebp)         ## 4-byte Spill
	calll	___assert_rtn
LBB0_5:
## BB#6:
	movl	-12(%ebp), %eax
	addl	$48, %esp
	popl	%esi
	popl	%edi
	popl	%ebp
	ret
Leh_func_end0:

	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.globl	__ZNSt3__111atomic_flag12test_and_setENS_12memory_orderE
	.weak_definition	__ZNSt3__111atomic_flag12test_and_setENS_12memory_orderE
	.align	4, 0x90
__ZNSt3__111atomic_flag12test_and_setENS_12memory_orderE: ## @_ZNSt3__111atomic_flag12test_and_setENS_12memory_orderE
Leh_func_begin1:
## BB#0:
	pushl	%ebp
Ltmp3:
	movl	%esp, %ebp
Ltmp4:
	pushl	%ebx
	subl	$20, %esp
Ltmp5:
	movl	12(%ebp), %eax
	movl	8(%ebp), %ecx
	movl	%ecx, -8(%ebp)
	movl	%eax, -12(%ebp)
	movl	-8(%ebp), %ecx
	movl	%esp, %edx
	movl	%eax, 4(%edx)
	movl	%ecx, (%edx)
	calll	__ZNVSt3__111atomic_flag12test_and_setENS_12memory_orderE
	movb	%al, -13(%ebp)          ## 1-byte Spill
	movzbl	%al, %eax
	movb	-13(%ebp), %bl          ## 1-byte Reload
	movb	%bl, -14(%ebp)          ## 1-byte Spill
	addl	$20, %esp
	popl	%ebx
	popl	%ebp
	ret
Leh_func_end1:

	.globl	__ZNVSt3__111atomic_flag12test_and_setENS_12memory_orderE
	.weak_definition	__ZNVSt3__111atomic_flag12test_and_setENS_12memory_orderE
	.align	4, 0x90
__ZNVSt3__111atomic_flag12test_and_setENS_12memory_orderE: ## @_ZNVSt3__111atomic_flag12test_and_setENS_12memory_orderE
## BB#0:
	pushl	%ebp
	movl	%esp, %ebp
	subl	$32, %esp
	movl	12(%ebp), %eax
	movl	8(%ebp), %ecx
	movl	%ecx, -8(%ebp)
	movl	%eax, -12(%ebp)
	movl	-8(%ebp), %ecx
	leal	-3(%eax), %edx
	cmpl	$3, %edx
	movl	%ecx, -20(%ebp)         ## 4-byte Spill
	movl	%eax, -24(%ebp)         ## 4-byte Spill
	jb	LBB2_2
## BB#4:
	movl	-24(%ebp), %eax         ## 4-byte Reload
	cmpl	$2, %eax
	ja	LBB2_3
## BB#1:
	movb	$-1, %al
	movl	-20(%ebp), %ecx         ## 4-byte Reload
	xchgb	%al, (%ecx)
	andb	$1, %al
	movb	%al, -1(%ebp)
	jmp	LBB2_3
LBB2_2:
	movb	$-1, %al
	movl	-20(%ebp), %ecx         ## 4-byte Reload
	xchgb	%al, (%ecx)
	andb	$1, %al
	movb	%al, -13(%ebp)
	#MEMBARRIER
	movb	-13(%ebp), %al
	andb	$1, %al
	movb	%al, -1(%ebp)
LBB2_3:
	movzbl	-1(%ebp), %eax
	movl	%eax, -28(%ebp)         ## 4-byte Spill
	movb	%al, %cl
	movl	-28(%ebp), %eax         ## 4-byte Reload
	movb	%cl, -29(%ebp)          ## 1-byte Spill
	addl	$32, %esp
	popl	%ebp
	ret

	.section	__TEXT,__cstring,cstring_literals
L___func__.main:                        ## @__func__.main
	.asciz	 "main"

	.align	4                       ## @.str
L_.str:
	.asciz	 "test_and_set.pass.cpp"

	.align	4                       ## @.str1
L_.str1:
	.asciz	 "f.test_and_set() == 0"

	.align	4                       ## @.str2
L_.str2:
	.asciz	 "f.test_and_set() == 1"

	.section	__TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame0:
Lsection_eh_frame0:
Leh_frame_common0:
Lset0 = Leh_frame_common_end0-Leh_frame_common_begin0 ## Length of Common Information Entry
	.long	Lset0
Leh_frame_common_begin0:
	.long	0                       ## CIE Identifier Tag
	.byte	1                       ## DW_CIE_VERSION
	.asciz	 "zR"                   ## CIE Augmentation
	.byte	1                       ## CIE Code Alignment Factor
	.byte	124                     ## CIE Data Alignment Factor
	.byte	8                       ## CIE Return Address Column
	.byte	1                       ## Augmentation Size
	.byte	16                      ## FDE Encoding = pcrel
	.byte	12                      ## DW_CFA_def_cfa
	.byte	5                       ## Register
	.byte	4                       ## Offset
	.byte	136                     ## DW_CFA_offset + Reg (8)
	.byte	1                       ## Offset
	.align	2
Leh_frame_common_end0:
	.globl	_main.eh
_main.eh:
Lset1 = Leh_frame_end0-Leh_frame_begin0 ## Length of Frame Information Entry
	.long	Lset1
Leh_frame_begin0:
Lset2 = Leh_frame_begin0-Leh_frame_common0 ## FDE CIE offset
	.long	Lset2
Ltmp6:                                  ## FDE initial location
	.long	Leh_func_begin0-Ltmp6
Lset3 = Leh_func_end0-Leh_func_begin0   ## FDE address range
	.long	Lset3
	.byte	0                       ## Augmentation size
	.byte	4                       ## DW_CFA_advance_loc4
Lset4 = Ltmp0-Leh_func_begin0
	.long	Lset4
	.byte	14                      ## DW_CFA_def_cfa_offset
	.byte	8                       ## Offset
	.byte	132                     ## DW_CFA_offset + Reg (4)
	.byte	2                       ## Offset
	.byte	4                       ## DW_CFA_advance_loc4
Lset5 = Ltmp1-Ltmp0
	.long	Lset5
	.byte	13                      ## DW_CFA_def_cfa_register
	.byte	4                       ## Register
	.byte	4                       ## DW_CFA_advance_loc4
Lset6 = Ltmp2-Ltmp1
	.long	Lset6
	.byte	134                     ## DW_CFA_offset + Reg (6)
	.byte	4                       ## Offset
	.byte	135                     ## DW_CFA_offset + Reg (7)
	.byte	3                       ## Offset
	.align	2
Leh_frame_end0:

	.globl	__ZNSt3__111atomic_flag12test_and_setENS_12memory_orderE.eh
	.weak_definition	__ZNSt3__111atomic_flag12test_and_setENS_12memory_orderE.eh
__ZNSt3__111atomic_flag12test_and_setENS_12memory_orderE.eh:
Lset7 = Leh_frame_end1-Leh_frame_begin1 ## Length of Frame Information Entry
	.long	Lset7
Leh_frame_begin1:
Lset8 = Leh_frame_begin1-Leh_frame_common0 ## FDE CIE offset
	.long	Lset8
Ltmp7:                                  ## FDE initial location
	.long	Leh_func_begin1-Ltmp7
Lset9 = Leh_func_end1-Leh_func_begin1   ## FDE address range
	.long	Lset9
	.byte	0                       ## Augmentation size
	.byte	4                       ## DW_CFA_advance_loc4
Lset10 = Ltmp3-Leh_func_begin1
	.long	Lset10
	.byte	14                      ## DW_CFA_def_cfa_offset
	.byte	8                       ## Offset
	.byte	132                     ## DW_CFA_offset + Reg (4)
	.byte	2                       ## Offset
	.byte	4                       ## DW_CFA_advance_loc4
Lset11 = Ltmp4-Ltmp3
	.long	Lset11
	.byte	13                      ## DW_CFA_def_cfa_register
	.byte	4                       ## Register
	.byte	4                       ## DW_CFA_advance_loc4
Lset12 = Ltmp5-Ltmp4
	.long	Lset12
	.byte	131                     ## DW_CFA_offset + Reg (3)
	.byte	3                       ## Offset
	.align	2
Leh_frame_end1:


.subsections_via_symbols
