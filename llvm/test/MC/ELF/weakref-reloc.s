// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the relocations point to the correct symbols. We used to get the
// symbol index wrong for weakrefs when creating _GLOBAL_OFFSET_TABLE_.

	.weakref	bar,foo
        call    zed@PLT
	call	bar

// CHECK:      # Symbol 0x00000004
// CHECK-NEXT: (('st_name', 0x00000009) # '_GLOBAL_OFFSET_TABLE_'
// CHECK-NEXT:  ('st_bind', 0x00000001)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000000)
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000005
// CHECK-NEXT: (('st_name', 0x00000001) # 'foo'
// CHECK-NEXT:  ('st_bind', 0x00000002)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000000)
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000006
// CHECK-NEXT: (('st_name', 0x00000005) # 'zed'
// CHECK-NEXT:  ('st_bind', 0x00000001)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000000)
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT: ),

// CHECK:      # Relocation 0x00000000
// CHECK-NEXT: (('r_offset', 0x00000001)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x00000004)
// CHECK-NEXT:  ('r_addend', 0xfffffffc)
// CHECK-NEXT: ),
// CHECK-NEXT: # Relocation 0x00000001
// CHECK-NEXT: (('r_offset', 0x00000006)
// CHECK-NEXT:  ('r_sym', 0x00000005)
// CHECK-NEXT:  ('r_type', 0x00000002)
// CHECK-NEXT:  ('r_addend', 0xfffffffc)
// CHECK-NEXT: ),
