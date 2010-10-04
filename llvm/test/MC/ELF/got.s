// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that this produces a R_X86_64_GOT32.

	movl	foo@GOT, %eax

// CHECK:      ('_relocations', [
// CHECK-NEXT:   # Relocation 0
// CHECK-NEXT:    (('r_offset',
// CHECK-NEXT:     ('r_sym',
// CHECK-NEXT:     ('r_type', 3)
// CHECK-NEXT:     ('r_addend',
// CHECK-NEXT:    ),
// CHECK-NEXT:   ])
