// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -t | FileCheck %s

// Test which symbols should be in the symbol table

        .long	.Lsym1
.Lsym2:
.Lsym3:
.Lsym4 = .Lsym2 - .Lsym3
        .long	.Lsym4

	.type	.Lsym5,@object
        .type   sym6,@object
        .long sym6

	.section	.rodata.str1.1,"aMS",@progbits,1
.Lsym7:
.Lsym8:

        .text
        movsd   .Lsym8(%rip), %xmm1

// CHECK:      Symbols [

// CHECK:        Symbol {
// CHECK:          Name: .Lsym8

// CHECK:        Symbol {
// CHECK:          Name: .Lsym1

// CHECK:        Symbol {
// CHECK:          Name: sym6
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT: ]
