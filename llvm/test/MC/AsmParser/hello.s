// RUN: llvm-mc -triple i386-unknown-unknown %s -o -
        
	.text
	.align	4,0x90
	.globl	_main
_main:
	pushl	%ebp
	movl	%esp, %ebp
	subl	$8, %esp
	call	"L1$pb"
"L1$pb":
	popl	%eax
	movl	$0, -4(%ebp)
	movl	%esp, %ecx
	leal	L_.str-"L1$pb"(%eax), %eax
	movl	%eax, (%ecx)
	call	_printf
	movl	$0, -4(%ebp)
	movl	-4(%ebp), %eax
	addl	$8, %esp
	popl	%ebp
	ret
	.subsections_via_symbols
	.cstring
L_.str:
	.asciz	"hello world!\n"

