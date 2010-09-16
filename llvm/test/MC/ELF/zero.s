// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s

.zero 4

// CHECK: ('sh_name', 1) # '.text'
// CHECK: ('sh_type', 1)
// CHECK: ('sh_flags', 6)
// CHECK: ('sh_addr', 0)
// CHECK: ('sh_offset', 64)
// CHECK: ('sh_size', 4)
// CHECK: ('sh_link', 0)
// CHECK: ('sh_info', 0)
// CHECK: ('sh_addralign', 4)
// CHECK: ('sh_entsize', 0)
// CHECK: ('_section_data', '00000000')
