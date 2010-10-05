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
// CHECK-NEXT:  # Symbol 0
// CHECK-NEXT:  (('st_name', 0) # ''
// CHECK:       # Symbol 1
// CHECK-NEXT:  (('st_name', 1) # '.Lsym8'
// CHECK:       # Symbol 2
// CHECK-NEXT:  (('st_name', 0) # ''
// CHECK:       # Symbol 3
// CHECK-NEXT:  (('st_name', 0) # ''
// CHECK:       # Symbol 4
// CHECK-NEXT:  (('st_name', 0) # ''
// CHECK:       # Symbol 5
// CHECK-NEXT:  (('st_name', 0) # ''
// CHECK:       # Symbol 6
// CHECK-NEXT:  (('st_name', 8) # '.Lsym1'
// CHECK:       # Symbol 7
// CHECK-NEXT:  (('st_name', 15) # 'sym6'
// CHECK-NEXT:   ('st_bind', 1)
// CHECK-NEXT:   ('st_type', 1)
// CHECK-NEXT:   ('st_other', 0)
// CHECK-NEXT:   ('st_shndx', 0)
// CHECK-NEXT:   ('st_value', 0)
// CHECK-NEXT:   ('st_size', 0)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
