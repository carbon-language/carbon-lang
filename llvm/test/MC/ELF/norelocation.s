// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

        call bar
bar:

// CHECK: ('sh_name', 0x1) # '.text'
// CHECK-NEXT: ('sh_type', 0x1)
// CHECK-NEXT: ('sh_flags', 0x6)
// CHECK-NEXT: ('sh_addr', 0x0)
// CHECK-NEXT: ('sh_offset', 0x40)
// CHECK-NEXT: ('sh_size', 0x5)
// CHECK-NEXT: ('sh_link', 0x0)
// CHECK-NEXT: ('sh_info', 0x0)
// CHECK-NEXT: ('sh_addralign', 0x4)
// CHECK-NEXT: ('sh_entsize', 0x0)
// CHECK-NEXT: ('_section_data', 'e8000000 00')
// CHECK-NOT: .rela.text
// CHECK: shstrtab
