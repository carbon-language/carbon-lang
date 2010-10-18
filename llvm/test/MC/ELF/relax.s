// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that we do a relaxation for foo but not for bar. Relaxing foo is
// probably not necessary, but matches what gnu as does.

// Also test that the relaxation done for foo uses the symbol, not section and
// offset.

bar:
.globl foo
foo:
        jmp bar
        jmp foo

// CHECK: ('sh_name', 0x1) # '.text'
// CHECK-NEXT: ('sh_type', 0x1)
// CHECK-NEXT: ('sh_flags', 0x6)
// CHECK-NEXT: ('sh_addr', 0x0)
// CHECK-NEXT: ('sh_offset', 0x40)
// CHECK-NEXT: ('sh_size', 0x7)
// CHECK-NEXT: ('sh_link', 0x0)
// CHECK-NEXT: ('sh_info', 0x0)
// CHECK-NEXT: ('sh_addralign', 0x4)
// CHECK-NEXT: ('sh_entsize', 0x0)
// CHECK-NEXT: ('_section_data', 'ebfee900 000000')

// CHECK:       # Symbol 0x5
// CHECK-NEXT: (('st_name', 0x5) # 'foo'

// CHECK: .rela.text
// CHECK: ('_relocations', [
// CHECK-NEXT: Relocation 0x0
// CHECK-NEXT:  (('r_offset', 0x3)
// CHECK-NEXT:   ('r_sym', 0x5)
// CHECK-NEXT:   ('r_type', 0x2)
// CHECK-NEXT:   ('r_addend', -0x4)
// CHECK-NEXT:  ),
// CHECK-NEXT: ])
