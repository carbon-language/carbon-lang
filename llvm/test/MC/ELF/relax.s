// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that we do not relax these.

bar:
.globl foo
foo:
        .set	zed,foo

        jmp bar
        jmp foo
        jmp zed

// CHECK: ('sh_name', 0x00000001) # '.text'
// CHECK-NEXT: ('sh_type', 0x00000001)
// CHECK-NEXT: ('sh_flags', 0x00000006)
// CHECK-NEXT: ('sh_addr', 0x00000000)
// CHECK-NEXT: ('sh_offset', 0x00000040)
// CHECK-NEXT: ('sh_size', 0x00000006)
// CHECK-NEXT: ('sh_link', 0x00000000)
// CHECK-NEXT: ('sh_info', 0x00000000)
// CHECK-NEXT: ('sh_addralign', 0x00000004)
// CHECK-NEXT: ('sh_entsize', 0x00000000)
// CHECK-NEXT: ('_section_data', 'ebfeebfc ebfa')

// CHECK:       # Symbol 6
// CHECK-NEXT: (('st_name', 0x00000005) # 'foo'
