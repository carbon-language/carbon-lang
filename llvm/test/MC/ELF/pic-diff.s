// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// CHECK:         # Symbol 0x00000005
// CHECK-NEXT:    (('st_name', 0x00000005) # 'baz'
// CHECK-NEXT:     ('st_bind', 0x00000001)
// CHECK-NEXT:     ('st_type', 0x00000000)
// CHECK-NEXT:     ('st_other', 0x00000000)
// CHECK-NEXT:     ('st_shndx', 0x00000000)
// CHECK-NEXT:     ('st_value', 0x0000000000000000)
// CHECK-NEXT:     ('st_size', 0x0000000000000000)
// CHECK-NEXT:    ),

// CHECK:       ('_relocations', [
// CHECK-NEXT:    # Relocation 0x00000000
// CHECK-NEXT:    (('r_offset', 0x0000000c)
// CHECK-NEXT:     ('r_sym', 0x00000005)
// CHECK-NEXT:     ('r_type', 0x00000002)
// CHECK-NEXT:     ('r_addend', 0x00000008)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ])

.zero 4
.data

.zero 1
.align 4
foo:
.zero 8
.long baz - foo
