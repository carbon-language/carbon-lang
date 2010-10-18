// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s


	.text

// Test that this produces a regular local symbol.
	.type	common1,@object
	.local	common1
	.comm	common1,1,1

// CHECK: ('st_name', 0x1) # 'common1'
// CHECK-NEXT: ('st_bind', 0x0)
// CHECK-NEXT: ('st_type', 0x1)
// CHECK-NEXT: ('st_other', 0x0)
// CHECK-NEXT: ('st_shndx',
// CHECK-NEXT: ('st_value', 0x0)
// CHECK-NEXT: ('st_size', 0x1)


// Same as common1, but with directives in a different order.
	.local	common2
	.type	common2,@object
	.comm	common2,1,1

// CHECK: ('st_name', 0x9) # 'common2'
// CHECK-NEXT: ('st_bind', 0x0)
// CHECK-NEXT: ('st_type', 0x1)
// CHECK-NEXT: ('st_other', 0x0)
// CHECK-NEXT: ('st_shndx',
// CHECK-NEXT: ('st_value', 0x1)
// CHECK-NEXT: ('st_size', 0x1)

// Test that without an explicit .local we produce a global.
	.type	common3,@object
	.comm	common3,4,4

// CHECK: ('st_name', 0x11) # 'common3'
// CHECK-NEXT: ('st_bind', 0x1)
// CHECK-NEXT: ('st_type', 0x1)
// CHECK-NEXT: ('st_other', 0x0)
// CHECK-NEXT: ('st_shndx', 0xfff2)
// CHECK-NEXT: ('st_value', 0x4)
// CHECK-NEXT: ('st_size', 0x4)


// Test that without an explicit .local we produce a global, even if the first
// occurrence is not in a directive.
	.globl	foo
	.type	foo,@function
foo:
	movsbl	common4+3(%rip), %eax


	.type	common4,@object
	.comm	common4,40,16

// CHECK: ('st_name', 0x1d) # 'common4'
// CHECK-NEXT: ('st_bind', 0x1)
// CHECK-NEXT: ('st_type', 0x1)
// CHECK-NEXT: ('st_other', 0x0)
// CHECK-NEXT: ('st_shndx', 0xfff2)
// CHECK-NEXT: ('st_value', 0x10)
// CHECK-NEXT: ('st_size', 0x28)
