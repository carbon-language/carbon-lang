// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that this produces a weak undefined symbol.

	.weak	foo
        .long   foo

//CHECK:       (('st_name', 1) # 'foo'
//CHECK-NEXT:   ('st_bind', 2)
//CHECK-NEXT:   ('st_type', 0)
//CHECK-NEXT:   ('st_other', 0)
//CHECK-NEXT:   ('st_shndx', 0)
//CHECK-NEXT:   ('st_value', 0)
//CHECK-NEXT:   ('st_size', 0)
