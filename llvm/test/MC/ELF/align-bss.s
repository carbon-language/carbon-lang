// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the bss section is correctly aligned

	.local	foo
	.comm	foo,2048,16

// CHECK:        ('sh_name', 13) # '.bss'
// CHECK-NEXT:   ('sh_type', 8)
// CHECK-NEXT:   ('sh_flags', 3)
// CHECK-NEXT:   ('sh_addr', 0)
// CHECK-NEXT:   ('sh_offset', 64)
// CHECK-NEXT:   ('sh_size', 2048)
// CHECK-NEXT:   ('sh_link', 0)
// CHECK-NEXT:   ('sh_info', 0)
// CHECK-NEXT:   ('sh_addralign', 16)
// CHECK-NEXT:   ('sh_entsize', 0)
