// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

foo:
bar = foo

        .globl	foo2
foo2 = bar2

foo3:
	.globl	bar3
bar3 = foo3

// Test that bar4 is also a function
        .type	foo4,@function
foo4:
bar4 = foo4

// CHECK:       # Symbol 0x1
// CHECK-NEXT:  (('st_name', 0x5) # 'bar'
// CHECK-NEXT:   ('st_bind', 0x0)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x0)
// CHECK-NEXT:   ('st_shndx', 0x1)
// CHECK-NEXT:   ('st_value', 0x0)
// CHECK-NEXT:   ('st_size', 0x0)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x2
// CHECK-NEXT: (('st_name', 0x1d) # 'bar4'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x2)
// CHECK-NEXT:  ('st_other', 0x0)
// CHECK-NEXT:  ('st_shndx', 0x1)
// CHECK-NEXT:  ('st_value', 0x0)
// CHECK-NEXT:  ('st_size', 0x0)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Symbol 0x3
// CHECK-NEXT:  (('st_name', 0x1) # 'foo'
// CHECK-NEXT:   ('st_bind', 0x0)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x0)
// CHECK-NEXT:   ('st_shndx', 0x1)
// CHECK-NEXT:   ('st_value', 0x0)
// CHECK-NEXT:   ('st_size', 0x0)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Symbol 0x4
// CHECK-NEXT:  (('st_name', 0xe) # 'foo3'
// CHECK-NEXT:   ('st_bind', 0x0)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x0)
// CHECK-NEXT:   ('st_shndx', 0x1)
// CHECK-NEXT:   ('st_value', 0x0)
// CHECK-NEXT:   ('st_size', 0x0)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x5
// CHECK-NEXT: (('st_name', 0x18) # 'foo4'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x2)
// CHECK-NEXT:  ('st_other', 0x0)
// CHECK-NEXT:  ('st_shndx', 0x1)
// CHECK-NEXT:  ('st_value', 0x0)
// CHECK-NEXT:  ('st_size', 0x0)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x6
// CHECK-NEXT: (('st_name', 0x0) # ''
// CHECK:       # Symbol 0x7
// CHECK-NEXT:  (('st_name', 0x0) # ''
// CHECK:       # Symbol 0x8
// CHECK-NEXT:  (('st_name', 0x0) # ''
// CHECK:       # Symbol 0x9
// CHECK-NEXT:  (('st_name', 0x13) # 'bar3'
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x0)
// CHECK-NEXT:   ('st_shndx', 0x1)
// CHECK-NEXT:   ('st_value', 0x0)
// CHECK-NEXT:   ('st_size', 0x0)
// CHECK:       # Symbol 0xa
// CHECK-NEXT:  (('st_name', 0x9) # 'bar2'
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x0)
// CHECK-NEXT:   ('st_shndx', 0x0)
// CHECK-NEXT:   ('st_value', 0x0)
// CHECK-NEXT:   ('st_size', 0x0)
