// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Mostly a test that this doesn't crash anymore.

// CHECK:        # Symbol 0x00000004
// CHECK-NEXT:    (('st_name', 0x00000001) # 'foo'
// CHECK-NEXT:     ('st_bind', 0x00000001)

	.size	foo, .Lbar-foo
