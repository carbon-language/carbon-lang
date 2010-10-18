// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s

.zero 4
.zero 1,42

// CHECK: ('sh_name', 0x1) # '.text'
// CHECK: ('sh_type', 0x1)
// CHECK: ('sh_flags', 0x6)
// CHECK: ('sh_addr', 0x0)
// CHECK: ('sh_offset', 0x40)
// CHECK: ('sh_size', 0x5)
// CHECK: ('sh_link', 0x0)
// CHECK: ('sh_info', 0x0)
// CHECK: ('sh_addralign', 0x4)
// CHECK: ('sh_entsize', 0x0)
// CHECK: ('_section_data', '00000000 2a')
