// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// CHECK:         # Symbol 5
// CHECK-NEXT:    (('st_name', 5) # 'baz'
// CHECK-NEXT:     ('st_bind', 1)
// CHECK-NEXT:     ('st_type', 0)
// CHECK-NEXT:     ('st_other', 0)
// CHECK-NEXT:     ('st_shndx', 0)
// CHECK-NEXT:     ('st_value', 0)
// CHECK-NEXT:     ('st_size', 0)
// CHECK-NEXT:    ),

// CHECK:       ('_relocations', [
// CHECK-NEXT:    # Relocation 0
// CHECK-NEXT:    (('r_offset', 12)
// CHECK-NEXT:     ('r_sym', 5)
// CHECK-NEXT:     ('r_type', 2)
// CHECK-NEXT:     ('r_addend', 8)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ])

.zero 4
foo:
.zero 8
.long baz - foo
