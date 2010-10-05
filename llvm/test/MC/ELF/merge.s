// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that PIC relocations with local symbols in a mergeable section are done
// with a reference to the symbol. Not sure if this is a linker limitation,
// but this matches the behavior of gas.

// Non-PIC relocations with 0 offset don't use the symbol.


        movsd   .Lfoo(%rip), %xmm1
        movl	$.Lfoo, %edi
        movl	$.Lfoo+2, %edi

        .section        .sec1,"aM",@progbits,16
.Lfoo:

// Section 4 is "sec1"
// CHECK: # Section 4
// CHECK-NEXT:  (('sh_name', 18) # '.sec1'

// Symbol number 1 is .Lfoo
// CHECK:      # Symbol 1
// CHECK-NEXT: (('st_name', 1) # '.Lfoo'

// Symbol number 5 is section 4
// CHECK:        # Symbol 5
// CHECK-NEXT:    (('st_name', 0) # ''
// CHECK-NEXT:     ('st_bind', 0)
// CHECK-NEXT:     ('st_type', 3)
// CHECK-NEXT:     ('st_other', 0)
// CHECK-NEXT:     ('st_shndx', 4)

// Relocation 0 refers to symbol 1
// CHECK:       ('_relocations', [
// CHECK-NEXT:   # Relocation 0
// CHECK-NEXT:   (('r_offset',
// CHECK-NEXT:    ('r_sym', 1)
// CHECK-NEXT:    ('r_type', 2
// CHECK-NEXT:    ('r_addend',
// CHECK-NEXT:   ),

// Relocation 1 refers to symbol 5
// CHECK-NEXT:  # Relocation 1
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym', 5)
// CHECK-NEXT:  ('r_type', 10)
// CHECK-NEXT:  ('r_addend',
// CHECK-NEXT: ),

// Relocation 2 refers to symbol 1
// CHECK-NEXT:   # Relocation 2
// CHECK-NEXT:   (('r_offset',
// CHECK-NEXT:    ('r_sym', 1)
// CHECK-NEXT:    ('r_type', 10
// CHECK-NEXT:    ('r_addend',
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
