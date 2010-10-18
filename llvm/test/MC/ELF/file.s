// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump | FileCheck %s

// Test that the STT_FILE symbol precedes the other local symbols.

.file "foo"
foa:
// CHECK:    # Symbol 0x1
// CHECK-NEXT:    (('st_name', 0x1) # 'foo'
// CHECK-NEXT:     ('st_bind', 0x0)
// CHECK-NEXT:     ('st_type', 0x4)
// CHECK-NEXT:     ('st_other', 0x0)
// CHECK-NEXT:     ('st_shndx', 0xfff1)
// CHECK-NEXT:     ('st_value', 0x0)
// CHECK-NEXT:     ('st_size', 0x0)
// CHECK-NEXT:    ),
// CHECK-NEXT:    # Symbol 0x2
// CHECK-NEXT:    (('st_name', 0x5) # 'foa'
// CHECK-NEXT:     ('st_bind', 0x0)
// CHECK-NEXT:     ('st_type', 0x0)
// CHECK-NEXT:     ('st_other', 0x0)
// CHECK-NEXT:     ('st_shndx', 0x1)
// CHECK-NEXT:     ('st_value', 0x0)
// CHECK-NEXT:     ('st_size', 0x0)
