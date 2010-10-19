// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump   | FileCheck %s

// Test that like gnu as we create text, data and bss by default. Also test
// that shstrtab, symtab and strtab are listed in that order.

// CHECK:      ('sh_name', 0x00000001) # '.text'
// CHECK-NEXT: ('sh_type', 0x00000001)
// CHECK-NEXT: ('sh_flags', 0x00000006)
// CHECK-NEXT: ('sh_addr', 0x00000000)
// CHECK-NEXT: ('sh_offset', 0x00000040)
// CHECK-NEXT: ('sh_size', 0x00000000)
// CHECK-NEXT: ('sh_link', 0x00000000)
// CHECK-NEXT: ('sh_info', 0x00000000)
// CHECK-NEXT: ('sh_addralign', 0x00000004)
// CHECK-NEXT: ('sh_entsize', 0x00000000)

// CHECK:      ('sh_name', 0x00000007) # '.data'
// CHECK-NEXT: ('sh_type', 0x00000001)
// CHECK-NEXT: ('sh_flags', 0x00000003)
// CHECK-NEXT: ('sh_addr', 0x00000000)
// CHECK-NEXT: ('sh_offset', 0x00000040)
// CHECK-NEXT: ('sh_size', 0x00000000)
// CHECK-NEXT: ('sh_link', 0x00000000)
// CHECK-NEXT: ('sh_info', 0x00000000)
// CHECK-NEXT: ('sh_addralign', 0x00000004)
// CHECK-NEXT: ('sh_entsize', 0x00000000)

// CHECK:      ('sh_name', 0x0000000d) # '.bss'
// CHECK-NEXT: ('sh_type', 0x00000008)
// CHECK-NEXT: ('sh_flags', 0x00000003)
// CHECK-NEXT: ('sh_addr', 0x00000000)
// CHECK-NEXT: ('sh_offset', 0x00000040)
// CHECK-NEXT: ('sh_size', 0x00000000)
// CHECK-NEXT: ('sh_link', 0x00000000)
// CHECK-NEXT: ('sh_info', 0x00000000)
// CHECK-NEXT: ('sh_addralign', 0x00000004)
// CHECK-NEXT: ('sh_entsize', 0x00000000)

// CHECK:      ('sh_name', 0x00000012) # '.shstrtab'
// CHECK-NEXT: ('sh_type', 0x00000003)
// CHECK-NEXT:    ('sh_flags', 0x00000000)
// CHECK-NEXT:    ('sh_addr', 0x00000000)
// CHECK-NEXT:    ('sh_offset', 0x00000040)
// CHECK-NEXT:    ('sh_size', 0x0000002c)
// CHECK-NEXT:    ('sh_link', 0x00000000)
// CHECK-NEXT:    ('sh_info', 0x00000000)
// CHECK-NEXT:    ('sh_addralign', 0x00000001)
// CHECK-NEXT:    ('sh_entsize', 0x00000000)

// CHECK: ('sh_name', 0x0000001c) # '.symtab'
// CHECK-NEXT:    ('sh_type', 0x00000002)
// CHECK-NEXT:    ('sh_flags', 0x00000000)
// CHECK-NEXT:    ('sh_addr', 0x00000000)
// CHECK-NEXT:    ('sh_offset',
// CHECK-NEXT:    ('sh_size', 0x00000060)
// CHECK-NEXT:    ('sh_link', 0x00000006)
// CHECK-NEXT:    ('sh_info', 0x00000004)
// CHECK-NEXT:    ('sh_addralign', 0x00000008)
// CHECK-NEXT:    ('sh_entsize', 0x00000018)

// CHECK: ('sh_name', 0x00000024) # '.strtab'
// CHECK-NEXT:    ('sh_type', 0x00000003)
// CHECK-NEXT:    ('sh_flags', 0x00000000)
// CHECK-NEXT:    ('sh_addr', 0x00000000)
// CHECK-NEXT:    ('sh_offset',
// CHECK-NEXT:    ('sh_size', 0x00000001)
// CHECK-NEXT:    ('sh_link', 0x00000000)
// CHECK-NEXT:    ('sh_info', 0x00000000)
// CHECK-NEXT:    ('sh_addralign', 0x00000001)
// CHECK-NEXT:    ('sh_entsize', 0x00000000)
