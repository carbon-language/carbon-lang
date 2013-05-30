// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r -t | FileCheck %s

// Test that this produces a R_X86_64_GOT32 and that we have an undefined
// reference to _GLOBAL_OFFSET_TABLE_.

        movl	foo@GOT, %eax
        movl	foo@GOTPCREL(%rip), %eax

// CHECK:      Relocations [
// CHECK:        Section ({{[^ ]+}}) .rela.text {
// CHECK-NEXT:       0x{{[^ ]+}} R_X86_64_GOT32 foo 0x{{[^ ]+}}
// CHECK-NEXT:       0x{{[^ ]+}} R_X86_64_GOTPCREL foo 0x{{[^ ]+}}
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK:        Symbol {
// CHECK:          Name: _GLOBAL_OFFSET_TABLE_
// CHECK-NEXT:     Value:
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Binding: Global
