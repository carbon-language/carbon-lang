// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the bss section is correctly aligned

	.local	foo
	.comm	foo,2048,16

// CHECK:        ('sh_name', 0xd) # '.bss'
// CHECK-NEXT:   ('sh_type', 0x8)
// CHECK-NEXT:   ('sh_flags', 0x3)
// CHECK-NEXT:   ('sh_addr', 0x0)
// CHECK-NEXT:   ('sh_offset', 0x40)
// CHECK-NEXT:   ('sh_size', 0x800)
// CHECK-NEXT:   ('sh_link', 0x0)
// CHECK-NEXT:   ('sh_info', 0x0)
// CHECK-NEXT:   ('sh_addralign', 0x10)
// CHECK-NEXT:   ('sh_entsize', 0x0)
