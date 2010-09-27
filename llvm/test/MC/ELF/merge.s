// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that relocations with local symbols in a mergeable section are done
// with a reference to the symbol. Not sure if this is a linker limitation,
// but this matches the behavior of gas.

        .section        .sec1,"aM",@progbits,16
.Lfoo:
        .text
        movsd   .Lfoo(%rip), %xmm1

// Symbol number 1 is .Lfoo

// CHECK:      # Symbol 1
// CHECK-NEXT: (('st_name', 1) # '.Lfoo'

// Relocation refers to symbol 1

// CHECK:       ('_relocations', [
// CHECK-NEXT:   # Relocation 0
// CHECK-NEXT:   (('r_offset',
// CHECK-NEXT:    ('r_sym', 1)
// CHECK-NEXT:    ('r_type',
// CHECK-NEXT:    ('r_addend',
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
