# The test verifies that correct DWARF directives are emitted when
# assembly files are instrumented.

# RUN: llvm-mc %s -triple=i386-unknown-linux-gnu -asm-instrumentation=address -asan-instrument-assembly | FileCheck %s

# CHECK-LABEL: swap_cfa_rbp
# CHECK: pushl %ebp
# CHECK-NOT: .cfi_adjust_cfa_offset 8
# CHECK: movl %ebp, %ebp
# CHECK: .cfi_remember_state
# CHECK: .cfi_def_cfa_register %ebp
# CHECK: popl %ebp
# CHECK: .cfi_restore_state
# CHECK-NOT: .cfi_adjust_cfa_offset -8
# CHECK: retl

	.text
	.globl	swap_cfa_rbp
	.type	swap_cfa_rbp,@function
swap_cfa_rbp:                                   # @swap_cfa_rbp
	.cfi_startproc
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register %ebp
	movl	8(%ebp), %eax
	movl	12(%ebp), %ecx
	movl	(%ecx), %ecx
	movl	%ecx, (%eax)
	popl	%ebp
	retl
	.cfi_endproc

# CHECK-LABEL: swap_cfa_rsp
# CHECK: pushl %ebp
# CHECK: .cfi_adjust_cfa_offset 4
# CHECK: movl %esp, %ebp
# CHECK: .cfi_remember_state
# CHECK: .cfi_def_cfa_register %ebp
# CHECK: popl %ebp
# CHECK: .cfi_restore_state
# CHECK: retl

	.globl	swap_cfa_rsp
	.type	swap_cfa_rsp,@function
swap_cfa_rsp:                                   # @swap_cfa_rsp
	.cfi_startproc
	pushl	%ebp
	.cfi_offset %ebp, 0
	movl	%esp, %ebp
	movl	8(%ebp), %eax
	movl	12(%ebp), %ecx
	movl	(%ecx), %ecx
	movl	%ecx, (%eax)
	popl	%ebp
	retl
	.cfi_endproc
