// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

        call bar
bar:

// CHECK: ('sh_name', 1) # '.text'
// CHECK-NEXT: ('sh_type', 1)
// CHECK-NEXT: ('sh_flags', 6)
// CHECK-NEXT: ('sh_addr', 0)
// CHECK-NEXT: ('sh_offset', 64)
// CHECK-NEXT: ('sh_size', 5)
// CHECK-NEXT: ('sh_link', 0)
// CHECK-NEXT: ('sh_info', 0)
// CHECK-NEXT: ('sh_addralign', 4)
// CHECK-NEXT: ('sh_entsize', 0)
// CHECK-NEXT: ('_section_data', 'e8000000 00')
// CHECK-NOT: .rela.text
// CHECK: shstrtab
