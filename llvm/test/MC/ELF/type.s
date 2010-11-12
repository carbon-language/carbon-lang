// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that both % and @ are accepted.
        .global foo
        .type foo,%function
foo:

        .global bar
        .type bar,@object
bar:

// CHECK:      # Symbol 0x00000004
// CHECK-NEXT: (('st_name', 0x00000005) # 'bar'
// CHECK-NEXT:  ('st_bind', 0x00000001)
// CHECK-NEXT:  ('st_type', 0x00000001)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000005
// CHECK-NEXT: (('st_name', 0x00000001) # 'foo'
// CHECK-NEXT:  ('st_bind', 0x00000001)
// CHECK-NEXT:  ('st_type', 0x00000002)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT: ),
