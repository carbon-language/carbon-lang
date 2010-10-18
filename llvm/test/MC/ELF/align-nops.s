// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump --dump-section-data | FileCheck %s

// Test that we get optimal nops in text
    .text
f0:
    .long 0
    .align  8, 0x90
    .long 0
    .align  8

// But not in another section
    .data
    .long 0
    .align  8, 0x90
    .long 0
    .align  8

// CHECK: (('sh_name', 0x1) # '.text'
// CHECK-NEXT:  ('sh_type', 0x1)
// CHECK-NEXT:  ('sh_flags', 0x6)
// CHECK-NEXT:  ('sh_addr',
// CHECK-NEXT:  ('sh_offset',
// CHECK-NEXT:  ('sh_size', 0x10)
// CHECK-NEXT:  ('sh_link', 0x0)
// CHECK-NEXT:  ('sh_info', 0x0)
// CHECK-NEXT:  ('sh_addralign', 0x8)
// CHECK-NEXT:  ('sh_entsize', 0x0)
// CHECK-NEXT:  ('_section_data', '00000000 0f1f4000 00000000 0f1f4000')

// CHECK: (('sh_name', 0x7) # '.data'
// CHECK-NEXT:  ('sh_type', 0x1)
// CHECK-NEXT:  ('sh_flags', 0x3)
// CHECK-NEXT:  ('sh_addr',
// CHECK-NEXT:  ('sh_offset',
// CHECK-NEXT:  ('sh_size', 0x10)
// CHECK-NEXT:  ('sh_link', 0x0)
// CHECK-NEXT:  ('sh_info', 0x0)
// CHECK-NEXT:  ('sh_addralign', 0x8)
// CHECK-NEXT:  ('sh_entsize', 0x0)
// CHECK-NEXT:  ('_section_data', '00000000 90909090 00000000 00000000')
