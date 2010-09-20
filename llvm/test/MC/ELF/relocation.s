// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that we produce a R_X86_64_32.

        .long   Lset1


// CHECK: # Relocation 0
// CHECK-NEXT:  (('r_offset', 0)
// CHECK-NEXT:   ('r_sym', 4)
// CHECK-NEXT:   ('r_type', 10)
// CHECK-NEXT:   ('r_addend', 0)
