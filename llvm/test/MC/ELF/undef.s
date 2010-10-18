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
// CHECK-NEXT:  # Symbol 0x0
// CHECK-NEXT:  (('st_name', 0x0) # ''
// CHECK:       # Symbol 0x1
// CHECK-NEXT:  (('st_name', 0xd) # '.Lsym8'
// CHECK:       # Symbol 0x2
// CHECK-NEXT:  (('st_name', 0x0) # ''
// CHECK:       # Symbol 0x3
// CHECK-NEXT:  (('st_name', 0x0) # ''
// CHECK:       # Symbol 0x4
// CHECK-NEXT:  (('st_name', 0x0) # ''
// CHECK:       # Symbol 0x5
// CHECK-NEXT:  (('st_name', 0x0) # ''
// CHECK:       # Symbol 0x6
// CHECK-NEXT:  (('st_name', 0x1) # '.Lsym1'
// CHECK:       # Symbol 0x7
// CHECK-NEXT:  (('st_name', 0x8) # 'sym6'
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x1)
// CHECK-NEXT:   ('st_other', 0x0)
// CHECK-NEXT:   ('st_shndx', 0x0)
// CHECK-NEXT:   ('st_value', 0x0)
// CHECK-NEXT:   ('st_size', 0x0)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
