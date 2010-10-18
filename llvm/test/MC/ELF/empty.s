// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump   | FileCheck %s

// Test that like gnu as we create text, data and bss by default. Also test
// that shstrtab, symtab and strtab are listed in that order.

// CHECK:      ('sh_name', 0x1) # '.text'
// CHECK-NEXT: ('sh_type', 0x1)
// CHECK-NEXT: ('sh_flags', 0x6)
// CHECK-NEXT: ('sh_addr', 0x0)
// CHECK-NEXT: ('sh_offset', 0x40)
// CHECK-NEXT: ('sh_size', 0x0)
// CHECK-NEXT: ('sh_link', 0x0)
// CHECK-NEXT: ('sh_info', 0x0)
// CHECK-NEXT: ('sh_addralign', 0x4)
// CHECK-NEXT: ('sh_entsize', 0x0)

// CHECK:      ('sh_name', 0x7) # '.data'
// CHECK-NEXT: ('sh_type', 0x1)
// CHECK-NEXT: ('sh_flags', 0x3)
// CHECK-NEXT: ('sh_addr', 0x0)
// CHECK-NEXT: ('sh_offset', 0x40)
// CHECK-NEXT: ('sh_size', 0x0)
// CHECK-NEXT: ('sh_link', 0x0)
// CHECK-NEXT: ('sh_info', 0x0)
// CHECK-NEXT: ('sh_addralign', 0x4)
// CHECK-NEXT: ('sh_entsize', 0x0)

// CHECK:      ('sh_name', 0xd) # '.bss'
// CHECK-NEXT: ('sh_type', 0x8)
// CHECK-NEXT: ('sh_flags', 0x3)
// CHECK-NEXT: ('sh_addr', 0x0)
// CHECK-NEXT: ('sh_offset', 0x40)
// CHECK-NEXT: ('sh_size', 0x0)
// CHECK-NEXT: ('sh_link', 0x0)
// CHECK-NEXT: ('sh_info', 0x0)
// CHECK-NEXT: ('sh_addralign', 0x4)
// CHECK-NEXT: ('sh_entsize', 0x0)

// CHECK:      ('sh_name', 0x12) # '.shstrtab'
// CHECK-NEXT: ('sh_type', 0x3)
// CHECK-NEXT:    ('sh_flags', 0x0)
// CHECK-NEXT:    ('sh_addr', 0x0)
// CHECK-NEXT:    ('sh_offset', 0x40)
// CHECK-NEXT:    ('sh_size', 0x2c)
// CHECK-NEXT:    ('sh_link', 0x0)
// CHECK-NEXT:    ('sh_info', 0x0)
// CHECK-NEXT:    ('sh_addralign', 0x1)
// CHECK-NEXT:    ('sh_entsize', 0x0)

// CHECK: ('sh_name', 0x1c) # '.symtab'
// CHECK-NEXT:    ('sh_type', 0x2)
// CHECK-NEXT:    ('sh_flags', 0x0)
// CHECK-NEXT:    ('sh_addr', 0x0)
// CHECK-NEXT:    ('sh_offset',
// CHECK-NEXT:    ('sh_size', 0x60)
// CHECK-NEXT:    ('sh_link', 0x6)
// CHECK-NEXT:    ('sh_info', 0x4)
// CHECK-NEXT:    ('sh_addralign', 0x8)
// CHECK-NEXT:    ('sh_entsize', 0x18)

// CHECK: ('sh_name', 0x24) # '.strtab'
// CHECK-NEXT:    ('sh_type', 0x3)
// CHECK-NEXT:    ('sh_flags', 0x0)
// CHECK-NEXT:    ('sh_addr', 0x0)
// CHECK-NEXT:    ('sh_offset',
// CHECK-NEXT:    ('sh_size', 0x1)
// CHECK-NEXT:    ('sh_link', 0x0)
// CHECK-NEXT:    ('sh_info', 0x0)
// CHECK-NEXT:    ('sh_addralign', 0x1)
// CHECK-NEXT:    ('sh_entsize', 0x0)
