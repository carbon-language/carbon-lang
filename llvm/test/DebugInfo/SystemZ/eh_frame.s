# RUN: llvm-mc -triple=s390x-linux-gnu -filetype=obj %s -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

	.text
	.globl	check_largest_class
	.align	4
	.type	check_largest_class,@function
check_largest_class:
	.cfi_startproc
	stmg	%r13, %r15, 104(%r15)
	.cfi_offset %r13, -56
	.cfi_offset %r14, -48
	.cfi_offset %r15, -40
	aghi	%r15, -160
	.cfi_def_cfa_offset 320
	lmg	%r13, %r15, 264(%r15)
	br	%r14
	.size	check_largest_class, .-check_largest_class
	.cfi_endproc

# The readelf rendering is:
#
# Contents of the .eh_frame section:
#
# 00000000 0000001c 00000000 CIE
#   Version:               1
#   Augmentation:          "zR"
#   Code alignment factor: 1
#   Data alignment factor: -8
#   Return address column: 14
#   Augmentation data:     1b
#
#   DW_CFA_def_cfa: r15 ofs 160
#   DW_CFA_nop
#   DW_CFA_nop
#   DW_CFA_nop
#
# 00000020 0000001c 00000024 FDE cie=00000000 pc=00000000..00000012
#   DW_CFA_advance_loc: 6 to 00000006
#   DW_CFA_offset: r13 at cfa-56
#   DW_CFA_offset: r14 at cfa-48
#   DW_CFA_offset: r15 at cfa-40
#   DW_CFA_advance_loc: 4 to 0000000a
#   DW_CFA_def_cfa_offset: 320
#   DW_CFA_nop
#   DW_CFA_nop
#   DW_CFA_nop
#   DW_CFA_nop
#
# CHECK: Contents of section .eh_frame:
# CHECK-NEXT: 0000 00000014 00000000 017a5200 01780e01  .........zR..x..
# CHECK-NEXT: 0010 1b0c0fa0 01000000 0000001c 0000001c  ................
# CHECK-NEXT: 0020 00000000 00000012 00468d07 8e068f05  .........F......
# CHECK-NEXT: 0030 440ec002 00000000                    D.......
