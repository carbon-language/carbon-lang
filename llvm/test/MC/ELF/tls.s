// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that all symbols are of type STT_TLS.

	leaq	foo1@TLSGD(%rip), %rdi
        leaq    foo2@GOTTPOFF(%rip), %rdi
        leaq    foo3@TLSLD(%rip), %rdi

	.section	.zed,"awT",@progbits
foobar:
	.long	43

// CHECK:      (('st_name', 0x00000010) # 'foobar'
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000006)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000005)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),

// CHECK:       # Symbol 0x00000007
// CHECK-NEXT:  (('st_name', 0x00000001) # 'foo1'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000006)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000008
// CHECK-NEXT:  (('st_name', 0x00000006) # 'foo2'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000006)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000009
// CHECK-NEXT:  (('st_name', 0x0000000b) # 'foo3'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000006)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
// CHECK-NEXT:  ),
