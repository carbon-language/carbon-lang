// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump   | FileCheck %s

// Test that we produce the group sections and that they are a the beginning
// of the file.

// CHECK:       # Section 0x00000001
// CHECK-NEXT:  (('sh_name', 0x00000026) # '.group'
// CHECK-NEXT:   ('sh_type', 0x00000011)
// CHECK-NEXT:   ('sh_flags', 0x00000000)
// CHECK-NEXT:   ('sh_addr', 0x00000000)
// CHECK-NEXT:   ('sh_offset', 0x00000040)
// CHECK-NEXT:   ('sh_size', 0x0000000c)
// CHECK-NEXT:   ('sh_link', 0x0000000c)
// CHECK-NEXT:   ('sh_info', 0x00000001)
// CHECK-NEXT:   ('sh_addralign', 0x00000004)
// CHECK-NEXT:   ('sh_entsize', 0x00000004)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Section 0x00000002
// CHECK-NEXT:  (('sh_name', 0x00000026) # '.group'
// CHECK-NEXT:   ('sh_type', 0x00000011)
// CHECK-NEXT:   ('sh_flags', 0x00000000)
// CHECK-NEXT:   ('sh_addr', 0x00000000)
// CHECK-NEXT:   ('sh_offset', 0x0000004c)
// CHECK-NEXT:   ('sh_size', 0x00000008)
// CHECK-NEXT:   ('sh_link', 0x0000000c)
// CHECK-NEXT:   ('sh_info', 0x00000002)
// CHECK-NEXT:   ('sh_addralign', 0x00000004)
// CHECK-NEXT:   ('sh_entsize', 0x00000004)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Section 0x00000003
// CHECK-NEXT:  (('sh_name', 0x00000026) # '.group'
// CHECK-NEXT:   ('sh_type', 0x00000011)
// CHECK-NEXT:   ('sh_flags', 0x00000000)
// CHECK-NEXT:   ('sh_addr', 0x00000000)
// CHECK-NEXT:   ('sh_offset', 0x00000054)
// CHECK-NEXT:   ('sh_size', 0x00000008)
// CHECK-NEXT:   ('sh_link', 0x0000000c)
// CHECK-NEXT:   ('sh_info', 0x0000000d)
// CHECK-NEXT:   ('sh_addralign', 0x00000004)
// CHECK-NEXT:   ('sh_entsize', 0x00000004)
// CHECK-NEXT:  ),

// Test that g1 and g2 are local, but g3 is an undefined global.

// CHECK:      # Symbol 0x00000001
// CHECK-NEXT: (('st_name', 0x00000001) # 'g1'
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000007)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000002
// CHECK-NEXT: (('st_name', 0x00000004) # 'g2'
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000002)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),

// CHECK:      # Symbol 0x0000000d
// CHECK-NEXT: (('st_name', 0x00000007) # 'g3'
// CHECK-NEXT:  ('st_bind', 0x00000001)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000000)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),


	.section	.foo,"axG",@progbits,g1,comdat
g1:
        nop

        .section	.bar,"axG",@progbits,g1,comdat
        nop

        .section	.zed,"axG",@progbits,g2,comdat
        nop

        .section	.baz,"axG",@progbits,g3,comdat
        .long g3
