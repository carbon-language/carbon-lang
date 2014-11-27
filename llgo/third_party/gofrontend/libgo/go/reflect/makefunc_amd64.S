# Copyright 2013 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# MakeFunc amd64 assembly code.

#include "config.h"

	.global	reflect.makeFuncStub

#ifdef __ELF__
	.type	reflect.makeFuncStub,@function
#endif

reflect.makeFuncStub:
.LFB1:

	# Store all the parameter registers in a struct that looks
	# like:
	# struct {
	#   rax uint64		// 0x0
	#   rdi uint64		// 0x8
	#   rsi uint64		// 0x10
	#   rdx uint64		// 0x18
	#   rcx uint64		// 0x20
	#   r8 uint64		// 0x28
	#   r9 uint64		// 0x30
	#   rsp uint64		// 0x38 Pointer to arguments on stack.
	#   xmm0 [2]uint64	// 0x40
	#   xmm1 [2]uint64	// 0x50
	#   xmm2 [2]uint64	// 0x60
	#   xmm3 [2]uint64	// 0x70
	#   xmm4 [2]uint64	// 0x80
	#   xmm5 [2]uint64	// 0x90
	#   xmm6 [2]uint64	// 0xa0
	#   xmm7 [2]uint64	// 0xb0
	# };

	pushq	%rbp
.LCFI0:
	movq	%rsp, %rbp
.LCFI1:

	subq	$0xc0, %rsp		# Space for struct on stack.

	movq	%rax, 0x0(%rsp)
	movq	%rdi, 0x8(%rsp)
	movq	%rsi, 0x10(%rsp)
	movq	%rdx, 0x18(%rsp)
	movq	%rcx, 0x20(%rsp)
	movq	%r8, 0x28(%rsp)
	movq	%r9, 0x30(%rsp)
	leaq	16(%rbp), %rax
	movq	%rax, 0x38(%rsp)
	movdqa	%xmm0, 0x40(%rsp)
	movdqa	%xmm1, 0x50(%rsp)
	movdqa	%xmm2, 0x60(%rsp)
	movdqa	%xmm3, 0x70(%rsp)
	movdqa	%xmm4, 0x80(%rsp)
	movdqa	%xmm5, 0x90(%rsp)
	movdqa	%xmm6, 0xa0(%rsp)
	movdqa	%xmm7, 0xb0(%rsp)

	/* For MakeFunc functions that call recover.  */
	movq	8(%rbp), %rdi
#ifdef __PIC__
	call	__go_makefunc_can_recover@PLT
#else
	call	__go_makefunc_can_recover
#endif

	# Get function type.
#ifdef __PIC__
	call	__go_get_closure@PLT
#else
	call	__go_get_closure
#endif
	movq	%rax, %rsi

	movq	%rsp, %rdi

#ifdef __PIC__
	call	reflect.MakeFuncStubGo@PLT
#else
	call	reflect.MakeFuncStubGo
#endif

	/* MakeFunc functions can no longer call recover.  */
#ifdef __PIC__
	call __go_makefunc_returning@PLT
#else
	call __go_makefunc_returning
#endif

	# The structure will be updated with any return values.  Load
	# all possible return registers before returning to the caller.

	movq	0x0(%rsp), %rax
	movq	0x18(%rsp), %rdx
	movq	0x8(%rsp), %rdi
	movq	0x10(%rsp), %rsi
	movdqa	0x40(%rsp), %xmm0
	movdqa	0x50(%rsp), %xmm1

	# long double values are returned on the floating point stack,
	# but we don't worry about that since Go doesn't have a long
	# double type.

	leave
.LCFI2:

	ret
.LFE1:

#ifdef __ELF__
	.size	reflect.makeFuncStub, . - reflect.makeFuncStub
#endif

#ifdef __ELF__
#ifdef HAVE_AS_X86_64_UNWIND_SECTION_TYPE
	.section	.eh_frame,"a",@unwind
#else
	.section	.eh_frame,"a",@progbits
#endif
.Lframe1:
	.long	.LECIE1-.LSCIE1	/* Length of Common Information Entry */
.LSCIE1:
	.long	0x0		/* CIE Identifier Tag */
	.byte	0x1		/* CIE Version */
	.ascii "zR\0"		/* CIE Augmentation */
	.uleb128 1		/* CIE Code Alignment Factor */
	.sleb128 -8		/* CIE Data Alignment Factor */
	.byte	0x10		/* CIE RA Column */
	.uleb128 1		/* Augmentation size */
	.byte	0x1b		/* FDE Encoding (pcrel sdata4) */
	.byte	0xc		/* DW_CFA_def_cfa, %rsp offset 8 */
	.uleb128 7
	.uleb128 8
	.byte	0x80+16		/* DW_CFA_offset, %rip offset 1*-8 */
	.uleb128 1
	.align 8
.LECIE1:
.LSFDE1:
	.long	.LEFDE1-.LASFDE1	/* FDE Length */
.LASFDE1:
	.long	.LASFDE1-.Lframe1	/* FDE CIE offset */
#if HAVE_AS_X86_PCREL
	.long	.LFB1-.			/* FDE initial location */
#else
	.long	.LFB1@rel
#endif
	.long	.LFE1-.LFB1		/* FDE address range */
	.uleb128 0x0			/* Augmentation size */
	.byte	0x4			/* DW_CFA_advance_loc4 */
	.long	.LCFI0-.LFB1
	.byte	0xe			/* DW_CFA_def_cfa_offset */
	.uleb128 16
	.byte	0x86			/* DW_CFA_offset, column 0x6 */
	.uleb128 2
	.byte	0x4			/* DW_CFA_advance_loc4 */
	.long	.LCFI1-.LCFI0
	.byte	0xd			/* DW_CFA_def_cfa_register */
	.uleb128 6
	.byte	0x2			/* DW_CFA_advance_loc1 */
	.byte	.LCFI2-.LCFI1
	.byte	0xc			/* DW_CFA_def_cfa */
	.uleb128 7
	.uleb128 8
	.align 8
.LEFDE1:
#endif /* __ELF__ */

#if defined(__ELF__) && defined(__linux__)
	.section	.note.GNU-stack,"",@progbits
	.section	.note.GNU-split-stack,"",@progbits
	.section	.note.GNU-no-split-stack,"",@progbits
#endif
