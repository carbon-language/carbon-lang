// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the alignment does contribute to the size of the section.

	.zero 4
	.align	8

// CHECK:      (('sh_name', 1) # '.text'
// CHECK-NEXT:  ('sh_type', 1)
// CHECK-NEXT:  ('sh_flags', 6)
// CHECK-NEXT:  ('sh_addr', 0)
// CHECK-NEXT:  ('sh_offset', 64)
// CHECK-NEXT:  ('sh_size', 8)
