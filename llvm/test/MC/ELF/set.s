// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that we accept .set of a symbol after it has been used in a statement.

        jmp foo
        .set foo, bar

// or a .quad

        .quad	foo2
	.set	foo2,bar2

// Test that there is an undefined reference to bar
// CHECK:      (('st_name', 0x00000001) # 'bar'
// CHECK-NEXT:  ('st_bind', 0x00000001)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000000)
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT: ),
