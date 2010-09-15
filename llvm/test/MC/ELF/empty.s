// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump   | FileCheck %s

// Test that like gnu as we create text, data and bss by default.

// CHECK:      ('sh_name', 1) # '.text'
// CHECK-NEXT: ('sh_type', 1)
// CHECK-NEXT: ('sh_flags', 6)
// CHECK-NEXT: ('sh_addr', 0)
// CHECK-NEXT: ('sh_offset', 64)
// CHECK-NEXT: ('sh_size', 0)
// CHECK-NEXT: ('sh_link', 0)
// CHECK-NEXT: ('sh_info', 0)
// CHECK-NEXT: ('sh_addralign', 4)
// CHECK-NEXT: ('sh_entsize', 0)

// CHECK:      ('sh_name', 7) # '.data'
// CHECK-NEXT: ('sh_type', 1)
// CHECK-NEXT: ('sh_flags', 3)
// CHECK-NEXT: ('sh_addr', 0)
// CHECK-NEXT: ('sh_offset', 64)
// CHECK-NEXT: ('sh_size', 0)
// CHECK-NEXT: ('sh_link', 0)
// CHECK-NEXT: ('sh_info', 0)
// CHECK-NEXT: ('sh_addralign', 4)
// CHECK-NEXT: ('sh_entsize', 0)

// CHECK:      ('sh_name', 13) # '.bss'
// CHECK-NEXT: ('sh_type', 8)
// CHECK-NEXT: ('sh_flags', 3)
// CHECK-NEXT: ('sh_addr', 0)
// CHECK-NEXT: ('sh_offset', 64)
// CHECK-NEXT: ('sh_size', 0)
// CHECK-NEXT: ('sh_link', 0)
// CHECK-NEXT: ('sh_info', 0)
// CHECK-NEXT: ('sh_addralign', 4)
// CHECK-NEXT: ('sh_entsize', 0)
