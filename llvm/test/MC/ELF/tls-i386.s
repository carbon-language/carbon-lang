// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that all symbols are of type STT_TLS.

        movl    foo1@NTPOFF(%eax), %eax
        movl    foo2@GOTNTPOFF(%eax), %eax
        movl    foo3@TLSGD(%eax), %eax
        movl    foo4@TLSLDM(%eax), %eax
        movl    foo5@TPOFF(%eax), %eax
        movl    foo6@DTPOFF(%eax), %eax
        movl    foo7@INDNTPOFF, %eax

// CHECK:       (('st_name', 0x00000001) # 'foo1'
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x00000006)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 6
// CHECK-NEXT:  (('st_name', 0x00000006) # 'foo2'
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x00000006)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 7
// CHECK-NEXT:  (('st_name', 0x0000000b) # 'foo3'
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x00000006)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 8
// CHECK-NEXT:  (('st_name', 0x00000010) # 'foo4'
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x00000006)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 9
// CHECK-NEXT:  (('st_name', 0x00000015) # 'foo5'
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x00000006)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 10
// CHECK-NEXT:  (('st_name', 0x0000001a) # 'foo6'
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x00000006)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 11
// CHECK-NEXT:  (('st_name', 0x0000001f) # 'foo7'
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x00000006)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:  ),
