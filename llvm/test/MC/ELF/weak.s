// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that this produces a weak undefined symbol.

	.weak	foo
        .long   foo

// And that bar is after all local symbols
        .weak bar
bar:

//CHECK:        # Symbol 4
//CHECK-NEXT:   (('st_name', 5) # 'bar'
//CHECK-NEXT:    ('st_bind', 2)
//CHECK-NEXT:    ('st_type', 0)
//CHECK-NEXT:    ('st_other', 0)
//CHECK-NEXT:    ('st_shndx', 1)
//CHECK-NEXT:    ('st_value', 0)
//CHECK-NEXT:    ('st_size', 0)
//CHECK-NEXT:   ),
//CHECK-NEXT:   # Symbol 5
//CHECK:       (('st_name', 1) # 'foo'
//CHECK-NEXT:   ('st_bind', 2)
//CHECK-NEXT:   ('st_type', 0)
//CHECK-NEXT:   ('st_other', 0)
//CHECK-NEXT:   ('st_shndx', 0)
//CHECK-NEXT:   ('st_value', 0)
//CHECK-NEXT:   ('st_size', 0)
//CHECK-NEXT:  ),
//CHECK-NEXT: ])
