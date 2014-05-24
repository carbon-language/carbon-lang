// RUN: llvm-mc -triple=arm64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck -check-prefix=OBJ %s

        b.eq somewhere

// OBJ:      Relocations [
// OBJ-NEXT:   Section (2) .rela.text {
// OBJ-NEXT:     0x0 R_AARCH64_CONDBR19 somewhere 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]
