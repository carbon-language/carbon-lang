// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test which symbols should be in the symbol table

        .long	.Lsym1
.Lsym2:
.Lsym3:
.Lsym4 = .Lsym2 - .Lsym3
        .long	.Lsym4

	.type	.Lsym5,@object
        .type   sym6,@object

	.section	.rodata.str1.1,"aMS",@progbits,1
.Lsym7:
.Lsym8:

        .text
        movsd   .Lsym8(%rip), %xmm1

// CHECK:      ('_symbols', [
// CHECK-NEXT:  # Symbol 0x00000000
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 0x00000001
// CHECK-NEXT:  (('st_name', 0x0000000d) # '.Lsym8'
// CHECK:       # Symbol 0x00000002
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 0x00000003
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 0x00000004
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 0x00000005
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 0x00000006
// CHECK-NEXT:  (('st_name', 0x00000001) # '.Lsym1'
// CHECK:       # Symbol 0x00000007
// CHECK-NEXT:  (('st_name', 0x00000008) # 'sym6'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000001)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
