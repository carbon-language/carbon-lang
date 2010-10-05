// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

foo:
bar = foo

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
