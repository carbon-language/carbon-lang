// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that these names are accepted.

.section	.note.GNU-stack,"",@progbits
.section	.note.GNU-stack2,"",%progbits
.section	.note.GNU-,"",@progbits
.section	-.note.GNU,"",@progbits

// CHECK: ('sh_name', 0x00000012) # '.note.GNU-stack'
// CHECK: ('sh_name', 0x00000022) # '.note.GNU-stack2'
// CHECK: ('sh_name', 0x00000033) # '.note.GNU-'
// CHECK: ('sh_name', 0x0000003e) # '-.note.GNU'

// Test that the defaults are used

.section	.init
.section	.fini
.section	.rodata
.section	zed, ""

// CHECK:      (('sh_name', 0x00000049) # '.init'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x00000006)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x00000050)
// CHECK-NEXT:  ('sh_size', 0x00000000)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x00000001)
// CHECK-NEXT:  ('sh_entsize', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Section 0x0000000b
// CHECK-NEXT: (('sh_name', 0x0000004f) # '.fini'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x00000006)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x00000050)
// CHECK-NEXT:  ('sh_size', 0x00000000)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x00000001)
// CHECK-NEXT:  ('sh_entsize', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Section 0x0000000c
// CHECK-NEXT: (('sh_name', 0x00000055) # '.rodata'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x00000002)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x00000050)
// CHECK-NEXT:  ('sh_size', 0x00000000)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x00000001)
// CHECK-NEXT:  ('sh_entsize', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Section 0x0000000d
// CHECK-NEXT: (('sh_name', 0x0000005d) # 'zed'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x00000000)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x00000050)
// CHECK-NEXT:  ('sh_size', 0x00000000)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x00000001)
// CHECK-NEXT:  ('sh_entsize', 0x00000000)
// CHECK-NEXT: ),

.section	.note.test,"",@note
// CHECK:       (('sh_name', 0x00000061) # '.note.test'
// CHECK-NEXT:   ('sh_type', 0x00000007)
// CHECK-NEXT:   ('sh_flags', 0x00000000)
// CHECK-NEXT:   ('sh_addr', 0x00000000)
// CHECK-NEXT:   ('sh_offset', 0x00000050)
// CHECK-NEXT:   ('sh_size', 0x00000000)
// CHECK-NEXT:   ('sh_link', 0x00000000)
// CHECK-NEXT:   ('sh_info', 0x00000000)
// CHECK-NEXT:   ('sh_addralign', 0x00000001)
// CHECK-NEXT:   ('sh_entsize', 0x00000000)
// CHECK-NEXT:  ),

// Test that we can parse these
foo:
bar:
.section        .text.foo,"axG",@progbits,foo,comdat
.section        .text.bar,"axMG",@progbits,42,bar,comdat

// Test that the default values are not used

.section .eh_frame,"a",@unwind

// CHECK:       (('sh_name', 0x00000080) # '.eh_frame'
// CHECK-NEXT:   ('sh_type', 0x70000001)
// CHECK-NEXT:   ('sh_flags', 0x00000002)
// CHECK-NEXT:   ('sh_addr', 0x00000000)
// CHECK-NEXT:   ('sh_offset', 0x00000050)
// CHECK-NEXT:   ('sh_size', 0x00000000)
// CHECK-NEXT:   ('sh_link', 0x00000000)
// CHECK-NEXT:   ('sh_info', 0x00000000)
// CHECK-NEXT:   ('sh_addralign', 0x00000001)
// CHECK-NEXT:   ('sh_entsize', 0x00000000)
// CHECK-NEXT:  ),

// Test that we handle the strings like gas
.section bar-"foo"
.section "foo"

// CHECK: ('sh_name', 0x0000008a) # 'bar-"foo"'
// CHECK: ('sh_name', 0x00000094) # 'foo'
