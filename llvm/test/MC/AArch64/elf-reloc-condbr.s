// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck -check-prefix=OBJ %s

        b.eq somewhere

// OBJ:      Relocations [
// OBJ-NEXT:   Section (1) .text {
// OBJ-NEXT:     0x0 R_AARCH64_CONDBR19 somewhere 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]
