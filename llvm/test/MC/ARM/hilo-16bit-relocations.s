@ RUN: llvm-mc %s -triple armv7-apple-darwin -show-encoding | FileCheck %s
        
_t:
        movw    r0, :lower16:(L_foo$non_lazy_ptr - (L1 + 8))
        movt    r0, :upper16:(L_foo$non_lazy_ptr - (L1 + 8))
L1:

@ CHECK: movw	r0, :lower16:(L_foo$non_lazy_ptr-(L1+8)) @ encoding: [A,A,0x00,0xe3]
@ CHECK:  @   fixup A - offset: 0, value: L_foo$non_lazy_ptr-(L1+8), kind: fixup_arm_movw_lo16_pcrel
@ CHECK: movt	r0, :upper16:(L_foo$non_lazy_ptr-(L1+8)) @ encoding: [A,A,0x40,0xe3]
@ CHECK:  @   fixup A - offset: 0, value: L_foo$non_lazy_ptr-(L1+8), kind: fixup_arm_movt_hi16_pcrel
        
        .comm	_foo,4,2

	.section	__DATA,__nl_symbol_ptr,non_lazy_symbol_pointers
	.align	2
L_foo$non_lazy_ptr:
	.indirect_symbol	_foo
	.long	0
        
.subsections_via_symbols
