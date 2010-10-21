// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that this produces a R_X86_64_PLT32. We produce a relocation with foo
// and gas with bar, but both should be OK as long as the type is correct.
        .globl foo
foo:
bar = foo
        call bar@PLT

// CHECK:       # Relocation 0
// CHECK-NEXT:  (('r_offset',
// CHECK-NEXT:   ('r_sym',
// CHECK-NEXT:   ('r_type', 0x00000004)
// CHECK-NEXT:   ('r_addend',
// CHECK-NEXT:  ),
// CHECK-NEXT: ])
