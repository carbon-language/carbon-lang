// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the alignment of rodata doesn't force a alignment of the
// previous section (.bss)

	nop
	.section	.rodata,"a",@progbits
	.align	8

// CHECK: # Section 0x3
// CHECK-NEXT:  (('sh_name', 0xd) # '.bss'
// CHECK-NEXT:   ('sh_type', 0x8)
// CHECK-NEXT:   ('sh_flags', 0x3)
// CHECK-NEXT:   ('sh_addr', 0x0)
// CHECK-NEXT:   ('sh_offset', 0x44)
// CHECK-NEXT:   ('sh_size', 0x0)
// CHECK-NEXT:   ('sh_link', 0x0)
// CHECK-NEXT:   ('sh_info', 0x0)
// CHECK-NEXT:   ('sh_addralign', 0x4)
// CHECK-NEXT:   ('sh_entsize', 0x0)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Section 0x4
// CHECK-NEXT:  (('sh_name', 0x12) # '.rodata'
// CHECK-NEXT:   ('sh_type', 0x1)
// CHECK-NEXT:   ('sh_flags', 0x2)
// CHECK-NEXT:   ('sh_addr', 0x0)
// CHECK-NEXT:   ('sh_offset', 0x48)
// CHECK-NEXT:   ('sh_size', 0x0)
// CHECK-NEXT:   ('sh_link', 0x0)
// CHECK-NEXT:   ('sh_info', 0x0)
// CHECK-NEXT:   ('sh_addralign', 0x8)
// CHECK-NEXT:   ('sh_entsize', 0x0)
