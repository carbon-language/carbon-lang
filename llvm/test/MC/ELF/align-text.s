// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the .text directive doesn't cause alignment.

        .zero 1
        .text
        .zero 1

// CHECK:      (('sh_name', 0x1) # '.text'
// CHECK-NEXT:  ('sh_type', 0x1)
// CHECK-NEXT:  ('sh_flags', 0x6)
// CHECK-NEXT:   ('sh_addr', 0x0)
// CHECK-NEXT:   ('sh_offset', 0x40)
// CHECK-NEXT:   ('sh_size', 0x2)
// CHECK-NEXT:   ('sh_link', 0x0)
// CHECK-NEXT:   ('sh_info', 0x0)
// CHECK-NEXT:   ('sh_addralign', 0x4)
// CHECK-NEXT:   ('sh_entsize', 0x0)
// CHECK-NEXT:  ),
