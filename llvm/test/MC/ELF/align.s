// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the alignment of rodata doesn't force a alignment of the
// previous section (.bss)

	nop
	.section	.rodata,"a",@progbits
	.align	8

// CHECK: # Section 3
// CHECK-NEXT:  (('sh_name', 13) # '.bss'
// CHECK-NEXT:   ('sh_type', 8)
// CHECK-NEXT:   ('sh_flags', 3)
// CHECK-NEXT:   ('sh_addr', 0)
// CHECK-NEXT:   ('sh_offset', 68)
// CHECK-NEXT:   ('sh_size', 0)
// CHECK-NEXT:   ('sh_link', 0)
// CHECK-NEXT:   ('sh_info', 0)
// CHECK-NEXT:   ('sh_addralign', 4)
// CHECK-NEXT:   ('sh_entsize', 0)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Section 4
// CHECK-NEXT:  (('sh_name', 18) # '.rodata'
// CHECK-NEXT:   ('sh_type', 1)
// CHECK-NEXT:   ('sh_flags', 2)
// CHECK-NEXT:   ('sh_addr', 0)
// CHECK-NEXT:   ('sh_offset', 72)
// CHECK-NEXT:   ('sh_size', 0)
// CHECK-NEXT:   ('sh_link', 0)
// CHECK-NEXT:   ('sh_info', 0)
// CHECK-NEXT:   ('sh_addralign', 8)
// CHECK-NEXT:   ('sh_entsize', 0)
