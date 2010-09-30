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

// CHECK: ('sh_name', 1) # '.text'
// CHECK-NEXT: ('sh_type', 1)
// CHECK-NEXT: ('sh_flags', 6)
// CHECK-NEXT: ('sh_addr', 0)
// CHECK-NEXT: ('sh_offset', 64)
// CHECK-NEXT: ('sh_size', 7)
// CHECK-NEXT: ('sh_link', 0)
// CHECK-NEXT: ('sh_info', 0)
// CHECK-NEXT: ('sh_addralign', 4)
// CHECK-NEXT: ('sh_entsize', 0)
// CHECK-NEXT: ('_section_data', 'ebfee900 000000')

// CHECK:       # Symbol 5
// CHECK-NEXT: (('st_name', 5) # 'foo'

// CHECK: .rela.text
// CHECK: ('_relocations', [
// CHECK-NEXT: Relocation 0
// CHECK-NEXT:  (('r_offset', 3)
// CHECK-NEXT:   ('r_sym', 5)
// CHECK-NEXT:   ('r_type', 2)
// CHECK-NEXT:   ('r_addend', -4)
// CHECK-NEXT:  ),
// CHECK-NEXT: ])
