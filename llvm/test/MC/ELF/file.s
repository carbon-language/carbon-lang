// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump | FileCheck %s

// Test that the STT_FILE symbol precedes the other local symbols.

.file "foo"
foa:
// CHECK:    # Symbol 1
// CHECK-NEXT:    (('st_name', 0x00000001) # 'foo'
// CHECK-NEXT:     ('st_bind', 0x00000000)
// CHECK-NEXT:     ('st_type', 0x00000004)
// CHECK-NEXT:     ('st_other', 0x00000000)
// CHECK-NEXT:     ('st_shndx', 0x0000fff1)
// CHECK-NEXT:     ('st_value', 0x0000000000000000)
// CHECK-NEXT:     ('st_size', 0x0000000000000000)
// CHECK-NEXT:    ),
// CHECK-NEXT:    # Symbol 2
// CHECK-NEXT:    (('st_name', 0x00000005) # 'foa'
// CHECK-NEXT:     ('st_bind', 0x00000000)
// CHECK-NEXT:     ('st_type', 0x00000000)
// CHECK-NEXT:     ('st_other', 0x00000000)
// CHECK-NEXT:     ('st_shndx', 0x00000001)
// CHECK-NEXT:     ('st_value', 0x0000000000000000)
// CHECK-NEXT:     ('st_size', 0x0000000000000000)
