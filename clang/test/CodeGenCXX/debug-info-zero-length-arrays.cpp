// RUN: %clang -target x86_64-unknown-unknown -fverbose-asm -g -O0 -S %s -o - | FileCheck %s
// <rdar://problem/12566646>

class A {
  int x[];
};
A a;

// CHECK:      Abbrev [3] 0x2d:0x3 DW_TAG_base_type
// CHECK-NEXT:   DW_AT_byte_size
// CHECK-NEXT:     DW_AT_encoding
// CHECK-NEXT:   Abbrev [4] 0x30:0xb DW_TAG_array_type
// CHECK-NEXT:     DW_AT_type
// CHECK-NEXT:     Abbrev [5] 0x35:0x5 DW_TAG_subrange_type
// CHECK-NEXT:       DW_AT_type
// CHECK-NEXT:   End Of Children Mark
