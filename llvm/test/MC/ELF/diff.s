// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

        .global zed
foo:
        nop
bar:
        nop
zed:
        mov zed+(bar-foo), %eax

// CHECK:       # Relocation 0x0
// CHECK-NEXT:  (('r_offset', 0x5)
// CHECK-NEXT:   ('r_sym', 0x6)
// CHECK-NEXT:   ('r_type', 0xb)
// CHECK-NEXT:   ('r_addend', 0x1)
