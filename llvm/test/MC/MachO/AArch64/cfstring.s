; RUN: llvm-mc -triple arm64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs | FileCheck %s

; Test that we produce an external relocation. There is no apparent need for it, but
; ld64 (241.9) produces a corrupt output if we don't.

// CHECK:      Relocations [
// CHECK-NEXT:   Section __data {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x0
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Extern: 1
// CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: Lfoo
// CHECK-NEXT:       Scattered: 0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

        .section        __DATA,__cfstring
Lfoo:

        .section        __DATA,__data
        .quad   Lfoo
