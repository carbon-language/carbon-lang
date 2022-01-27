// REQUIRES: ppc

// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/ppc64-func-global-entry.s -o %t2.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/ppc64-func-local-entry.s -o %t3.o
// RUN: ld.lld %t.o %t2.o %t3.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/ppc64-func-global-entry.s -o %t2.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/ppc64-func-local-entry.s -o %t3.o
// RUN: ld.lld %t.o %t2.o %t3.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

	.text
	.abiversion 2
	.globl	_start                    # -- Begin function _start
	.p2align	4
	.type	_start,@function
_start:                                   # @_start
.Lfunc_begin0:
.Lfunc_gep0:
	addis 2, 12, .TOC.-.Lfunc_gep0@ha
	addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
	.localentry	_start, .Lfunc_lep0-.Lfunc_gep0
# %bb.0:                                # %entry
	mflr 0
	std 0, 16(1)
	stdu 1, -48(1)
	li 3, 1
	li 4, 1
	std 30, 32(1)                   # 8-byte Folded Spill
	bl foo_external_same
	nop
	mr 30, 3
	li 3, 2
	li 4, 2
	bl foo_external_diff
	nop
	addis 4, 2, .LC0@toc@ha
	add 3, 3, 30
	ld 30, 32(1)                    # 8-byte Folded Reload
	ld 4, .LC0@toc@l(4)
	lwz 4, 0(4)
	add 3, 3, 4
	extsw 3, 3
	addi 1, 1, 48
	ld 0, 16(1)
	li 0, 1
	sc
	.long	0
	.quad	0
.Lfunc_end0:
	.size	_start, .Lfunc_end0-.Lfunc_begin0
                                        # -- End function
	.section	.toc,"aw",@progbits
.LC0:
	.tc glob[TC],glob
	.type	glob,@object            # @glob
	.data
	.globl	glob
	.p2align	2
glob:
	.long	10                      # 0xa
	.size	glob, 4

# Check that foo_external_diff has a global entry point and we branch to
# foo_external_diff+8. Also check that foo_external_same has no global entry
# point and we branch to start of foo_external_same.

// CHECK-LABEL: <_start>:
// CHECK:         100101f0: bl 0x10010280
// CHECK:         10010204: bl 0x10010258
// CHECK-LABEL: <foo_external_diff>:
// CHECK-NEXT:    10010250: addis 2, 12, 2
// CHECK-NEXT:    10010254: addi 2, 2, -32696
// CHECK-NEXT:    10010258: addis 5, 2, 1
// CHECK-LABEL: <foo_external_same>:
// CHECK-NEXT:    10010280: add 3, 4, 3
