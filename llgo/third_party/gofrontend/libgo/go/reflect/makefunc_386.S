/* Copyright 2013 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.

   MakeFunc 386 assembly code.  */

#include "config.h"

	.globl reflect.makeFuncStub

#ifdef __ELF__
	.type reflect.makeFuncStub,@function
#endif

reflect.makeFuncStub:
.LFB1:

	/* Go does not provide any equivalent to the regparm function
	   attribute, so on Go we do not need to worry about passing
	   parameters in registers.  We just pass a pointer to the
	   arguments on the stack.
	
	   We do need to pick up the return values, though, so we pass
	   a pointer to a struct that looks like this.
	   struct {
	     esp uint32		// 0x0
	     eax uint32		// 0x4
	     st0 float64	// 0x8
	     sr  bool		// 0x10
	     sf  bool		// 0x11
	   }
	   The sr field is set by the function to a non-zero value if
	   the function takes a struct hidden pointer that must be
	   popped off the stack.  */

	pushl	%ebp
.LCFI0:
	movl	%esp, %ebp
.LCFI1:
	pushl	%ebx		/* In case this is PIC.  */
	subl	$36, %esp	/* Enough for args and to align stack.  */
.LCFI2:

#ifdef __PIC__
	call	__x86.get_pc_thunk.bx
	addl	$_GLOBAL_OFFSET_TABLE_, %ebx
#endif

	leal	8(%ebp), %eax	/* Set esp field in struct.  */
	movl	%eax, -24(%ebp)

	/* For MakeFunc functions that call recover.  */
	movl	4(%ebp), %eax
	movl	%eax, (%esp)
#ifdef __PIC__
	call	__go_makefunc_can_recover@PLT
#else
	call	__go_makefunc_can_recover
#endif

#ifdef __PIC__
	call	__go_get_closure@PLT
#else
	call	__go_get_closure
#endif

	movl	%eax, 4(%esp)

	leal	-24(%ebp), %eax
	movl	%eax, (%esp)

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

	/* Set return registers.  */

	movl	-20(%ebp), %eax

	cmpb	$0, -7(%ebp)
	je	2f

	fldl	-16(%ebp)

#ifdef __SSE2__
	/* In case we are compiling with -msseregparm.  This won't work
	   correctly if only SSE1 is supported, but that seems unlikely.  */
	movsd	-16(%ebp), %xmm0
#endif

2:
	movb	-8(%ebp), %dl

	addl	$36, %esp
	popl	%ebx
.LCFI3:
	popl	%ebp
.LCFI4:

	testb	%dl,%dl
	jne	1f
	ret
1:
	ret	$4
.LFE1:
#ifdef __ELF__
	.size	reflect.makeFuncStub, . - reflect.makeFuncStub
#endif

#ifdef __PIC__
#ifdef HAVE_AS_COMDAT_GAS
	.section	.text.__x86.get_pc_thunk.bx,"axG",@progbits,__x86.get_pc_thunk.bx,comdat
#else
	/* Sun as needs a different syntax.  */
	.section	.text.__x86.get_pc_thunk.bx%__x86.get_pc_thunk.bx,"ax",@progbits
	.group		__x86.get_pc_thunk.bx,.text.__x86.get_pc_thunk.bx%__x86.get_pc_thunk.bx,#comdat
#endif
	.globl	__x86.get_pc_thunk.bx
	.hidden	__x86.get_pc_thunk.bx
#ifdef __ELF__
	.type	__x86.get_pc_thunk.bx, @function
#endif
__x86.get_pc_thunk.bx:
.LFB2:
	movl	(%esp), %ebx
	ret
.LFE2:
#ifdef __ELF__
	.size	__x86.get_pc_thunk.bx, . - __x86.get_pc_thunk.bx
#endif
#endif

#ifdef __ELF__
#if defined __PIC__
# if defined __sun__ && defined __svr4__
/* 32-bit Solaris 2/x86 uses datarel encoding for PIC.  GNU ld before 2.22
   doesn't correctly sort .eh_frame_hdr with mixed encodings, so match this.  */
#  define FDE_ENCODING		0x30	/* datarel */
#  define FDE_ENCODE(X)		X@GOTOFF
# else
#  define FDE_ENCODING		0x1b	/* pcrel sdata4 */
#  if defined HAVE_AS_X86_PCREL
#   define FDE_ENCODE(X)	X-.
#  else
#   define FDE_ENCODE(X)	X@rel
#  endif
# endif
#else
# define FDE_ENCODING		0	/* absolute */
# define FDE_ENCODE(X)		X
#endif

	.section	.eh_frame,EH_FRAME_FLAGS,@progbits
.Lframe1:
	.long	.LECIE1-.LSCIE1	/* Length of Common Information Entry */
.LSCIE1:
	.long	0x0	/* CIE Identifier Tag */
	.byte	0x1	/* CIE Version */
	.ascii "zR\0"	/* CIE Augmentation */
	.byte	0x1	/* .uleb128 0x1; CIE Code Alignment Factor */
	.byte	0x7c	/* .sleb128 -4; CIE Data Alignment Factor */
	.byte	0x8	/* CIE RA Column */
	.byte	0x1	/* .uleb128 0x1; Augmentation size */
	.byte	FDE_ENCODING
	.byte	0xc	/* DW_CFA_def_cfa */
	.byte	0x4	/* .uleb128 0x4 */
	.byte	0x4	/* .uleb128 0x4 */
	.byte	0x88	/* DW_CFA_offset, column 0x8 */
	.byte	0x1	/* .uleb128 0x1 */
	.align 4
.LECIE1:
.LSFDE1:
	.long	.LEFDE1-.LASFDE1	/* FDE Length */
.LASFDE1:
	.long	.LASFDE1-.Lframe1	/* FDE CIE offset */
	.long	FDE_ENCODE(.LFB1)	/* FDE initial location */
	.long	.LFE1-.LFB1		/* FDE address range */
	.byte	0x0	/* .uleb128 0x0; Augmentation size */
	.byte	0x4	/* DW_CFA_advance_loc4 */
	.long	.LCFI0-.LFB1
	.byte	0xe	/* DW_CFA_def_cfa_offset */
	.byte	0x8	/* .uleb128 0x8 */
	.byte	0x85	/* DW_CFA_offset, column 0x5 */
	.byte	0x2	/* .uleb128 0x2 */
	.byte	0x4	/* DW_CFA_advance_loc4 */
	.long	.LCFI1-.LCFI0
	.byte	0xd	/* DW_CFA_def_cfa_register */
	.byte	0x5	/* .uleb128 0x5 */
	.byte	0x4	/* DW_CFA_advance_loc4 */
	.long	.LCFI2-.LCFI1
	.byte	0x83	/* .DW_CFA_offset, column 0x3 */
	.byte	0x3	/* .uleb128 0x3 */
	.byte	0x4	/* DW_CFA_advance_loc4 */
	.long	.LCFI3-.LCFI2
	.byte	0xc3	/* DW_CFA_restore, column 0x3 */
	.byte	0x4	/* DW_CFA_advance_loc4 */
	.long	.LCFI4-.LCFI3
	.byte	0xc5	/* DW_CFA_restore, column 0x5 */
	.byte	0xc	/* DW_CFA_def_cfa */
	.byte	0x4	/* .uleb128 0x4 */
	.byte	0x4	/* .uleb128 0x4 */
	.align 4
.LEFDE1:
#ifdef __PIC__
.LSFDE2:
	.long	.LEFDE2-.LASFDE2	/* FDE Length */
.LASFDE2:
	.long	.LASFDE2-.Lframe1	/* FDE CIE offset */
	.long	FDE_ENCODE(.LFB2)	/* FDE initial location */
	.long	.LFE2-.LFB2		/* FDE address range */
	.byte	0x0	/* .uleb128 0x0; Augmentation size */
	.align 4
.LEFDE2:
#endif /* __PIC__ */
#endif /* __ELF__ */

#if defined(__ELF__) && defined(__linux__)
	.section	.note.GNU-stack,"",@progbits
	.section	.note.GNU-split-stack,"",@progbits
	.section	.note.GNU-no-split-stack,"",@progbits
#endif
