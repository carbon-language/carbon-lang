// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

foo:
bar = foo

        .globl	foo2
foo2 = bar2

foo3:
	.globl	bar3
bar3 = foo3

// CHECK:       # Symbol 1
// CHECK-NEXT:  (('st_name', 5) # 'bar'
// CHECK-NEXT:   ('st_bind', 0)
// CHECK-NEXT:   ('st_type', 0)
// CHECK-NEXT:   ('st_other', 0)
// CHECK-NEXT:   ('st_shndx', 1)
// CHECK-NEXT:   ('st_value', 0)
// CHECK-NEXT:   ('st_size', 0)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 2
// CHECK-NEXT:  (('st_name', 1) # 'foo'
// CHECK-NEXT:   ('st_bind', 0)
// CHECK-NEXT:   ('st_type', 0)
// CHECK-NEXT:   ('st_other', 0)
// CHECK-NEXT:   ('st_shndx', 1)
// CHECK-NEXT:   ('st_value', 0)
// CHECK-NEXT:   ('st_size', 0)
// CHECK:       # Symbol 3
// CHECK-NEXT:  (('st_name', 9) # 'foo3'
// CHECK-NEXT:   ('st_bind', 0)
// CHECK-NEXT:   ('st_type', 0)
// CHECK-NEXT:   ('st_other', 0)
// CHECK-NEXT:   ('st_shndx', 1)
// CHECK-NEXT:   ('st_value', 0)
// CHECK-NEXT:   ('st_size', 0)
// CHECK:       # Symbol 4
// CHECK-NEXT:  (('st_name', 0) # ''
// CHECK:       # Symbol 5
// CHECK-NEXT:  (('st_name', 0) # ''
// CHECK:       # Symbol 6
// CHECK-NEXT:  (('st_name', 0) # ''
// CHECK:       # Symbol 7
// CHECK-NEXT:  (('st_name', 24) # 'bar3'
// CHECK-NEXT:   ('st_bind', 1)
// CHECK-NEXT:   ('st_type', 0)
// CHECK-NEXT:   ('st_other', 0)
// CHECK-NEXT:   ('st_shndx', 1)
// CHECK-NEXT:   ('st_value', 0)
// CHECK-NEXT:   ('st_size', 0)
// CHECK:       # Symbol 8
// CHECK-NEXT:  (('st_name', 19) # 'bar2'
// CHECK-NEXT:   ('st_bind', 1)
// CHECK-NEXT:   ('st_type', 0)
// CHECK-NEXT:   ('st_other', 0)
// CHECK-NEXT:   ('st_shndx', 0)
// CHECK-NEXT:   ('st_value', 0)
// CHECK-NEXT:   ('st_size', 0)
