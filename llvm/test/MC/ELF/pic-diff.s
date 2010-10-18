// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// CHECK:         # Symbol 0x5
// CHECK-NEXT:    (('st_name', 0x5) # 'baz'
// CHECK-NEXT:     ('st_bind', 0x1)
// CHECK-NEXT:     ('st_type', 0x0)
// CHECK-NEXT:     ('st_other', 0x0)
// CHECK-NEXT:     ('st_shndx', 0x0)
// CHECK-NEXT:     ('st_value', 0x0)
// CHECK-NEXT:     ('st_size', 0x0)
// CHECK-NEXT:    ),

// CHECK:       ('_relocations', [
// CHECK-NEXT:    # Relocation 0x0
// CHECK-NEXT:    (('r_offset', 0xc)
// CHECK-NEXT:     ('r_sym', 0x5)
// CHECK-NEXT:     ('r_type', 0x2)
// CHECK-NEXT:     ('r_addend', 0x8)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ])

.zero 4
.data

.zero 1
.align 4
foo:
.zero 8
.long baz - foo
