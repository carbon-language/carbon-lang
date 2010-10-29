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

        .long foo2
// CHECK:       # Symbol 0x00000001
// CHECK-NEXT:  (('st_name', 0x00000005) # 'bar'
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000002
// CHECK-NEXT: (('st_name', 0x0000001d) # 'bar4'
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000002)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Symbol 0x00000003
// CHECK-NEXT:  (('st_name', 0x00000001) # 'foo'
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Symbol 0x00000004
// CHECK-NEXT:  (('st_name', 0x0000000e) # 'foo3'
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000005
// CHECK-NEXT: (('st_name', 0x00000018) # 'foo4'
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000002)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000006
// CHECK-NEXT: (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 0x00000007
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 0x00000008
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 0x00000009
// CHECK-NEXT:  (('st_name', 0x00000013) # 'bar3'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK:       # Symbol 0x0000000a
// CHECK-NEXT:  (('st_name', 0x00000009) # 'bar2'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
