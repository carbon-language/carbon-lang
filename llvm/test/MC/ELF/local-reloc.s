// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that relocations with local symbols are represented as relocations
// with the section. They should be equivalent, but gas behaves like this.

	movl	foo, %r14d
foo:

// Section number 1 is .text
// CHECK:        # Section 1
// CHECK-next:  (('sh_name', 1) # '.text'

// Symbol number 2 is section number 1
// CHECK:    # Symbol 2
// CHECK-NEXT:    (('st_name', 0) # ''
// CHECK-NEXT:     ('st_bind', 0)
// CHECK-NEXT:     ('st_type', 3)
// CHECK-NEXT:     ('st_other', 0)
// CHECK-NEXT:     ('st_shndx', 1)
// CHECK-NEXT:     ('st_value', 0)
// CHECK-NEXT:     ('st_size', 0)

// Relocation refers to symbol number 2
// CHECK:      ('_relocations', [
// CHECK-NEXT:  # Relocation 0
// CHECK-NEXT:   (('r_offset',
// CHECK-NEXT:    ('r_sym', 2)
// CHECK-NEXT:    ('r_type',
// CHECK-NEXT:    ('r_addend',
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
