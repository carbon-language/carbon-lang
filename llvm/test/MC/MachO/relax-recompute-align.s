// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

// FIXME: This is a horrible way of checking the output, we need an llvm-mc
// based 'otool'.

// FIXME: PR8467.
// There is an unnecessary relaxation here. After the first jmp slides,
// the .align size could be recomputed so that the second jump will be in range
// for a 1-byte jump. For performance reasons, this is not currently done.

// CHECK:  # Section 0
// CHECK: (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:  ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:  ('address', 0)
// CHECK:  ('size', 322)
// CHECK:  ('offset', 324)
// CHECK:  ('alignment', 4)
// CHECK:  ('reloc_offset', 0)
// CHECK:  ('num_reloc', 0)
// CHECK:  ('flags', 0x80000400)
// CHECK:  ('reserved1', 0)
// CHECK:  ('reserved2', 0)
// CHECK: ),

L0:
        .space 0x8a, 0x90
	jmp	L0
        .space (0xb3 - 0x8f), 0x90
	jle	L2
        .space (0xcd - 0xb5), 0x90
	.align	4, 0x90
L1:
        .space (0x130 - 0xd0),0x90
	jl	L1
L2:

.zerofill __DATA,__bss,_sym,4,2
