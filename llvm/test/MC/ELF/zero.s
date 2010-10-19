// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s

.zero 4
.zero 1,42

// CHECK: ('sh_name', 0x00000001) # '.text'
// CHECK: ('sh_type', 0x00000001)
// CHECK: ('sh_flags', 0x00000006)
// CHECK: ('sh_addr', 0x00000000)
// CHECK: ('sh_offset', 0x00000040)
// CHECK: ('sh_size', 0x00000005)
// CHECK: ('sh_link', 0x00000000)
// CHECK: ('sh_info', 0x00000000)
// CHECK: ('sh_addralign', 0x00000004)
// CHECK: ('sh_entsize', 0x00000000)
// CHECK: ('_section_data', '00000000 2a')
