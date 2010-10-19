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

// CHECK: ('sh_name', 0x00000001) # '.text'
// CHECK-NEXT: ('sh_type', 0x00000001)
// CHECK-NEXT: ('sh_flags', 0x00000006)
// CHECK-NEXT: ('sh_addr', 0x00000000)
// CHECK-NEXT: ('sh_offset', 0x00000040)
// CHECK-NEXT: ('sh_size', 0x00000007)
// CHECK-NEXT: ('sh_link', 0x00000000)
// CHECK-NEXT: ('sh_info', 0x00000000)
// CHECK-NEXT: ('sh_addralign', 0x00000004)
// CHECK-NEXT: ('sh_entsize', 0x00000000)
// CHECK-NEXT: ('_section_data', 'ebfee900 000000')

// CHECK:       # Symbol 0x00000005
// CHECK-NEXT: (('st_name', 0x00000005) # 'foo'

// CHECK: .rela.text
// CHECK: ('_relocations', [
// CHECK-NEXT: Relocation 0x00000000
// CHECK-NEXT:  (('r_offset', 0x00000003)
// CHECK-NEXT:   ('r_sym', 0x00000005)
// CHECK-NEXT:   ('r_type', 0x00000002)
// CHECK-NEXT:   ('r_addend', 0xfffffffc)
// CHECK-NEXT:  ),
// CHECK-NEXT: ])
