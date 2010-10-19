// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

        .global zed
foo:
        nop
bar:
        nop
zed:
        mov zed+(bar-foo), %eax

// CHECK:       # Relocation 0
// CHECK-NEXT:  (('r_offset', 5)
// CHECK-NEXT:   ('r_sym', 6)
// CHECK-NEXT:   ('r_type', 11)
// CHECK-NEXT:   ('r_addend', 1)
