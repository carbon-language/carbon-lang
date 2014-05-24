// RUN: llvm-mc -triple=arm64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck -check-prefix=OBJ %s

        add x2, x3, #:lo12:some_label

// OBJ:      Relocations [
// OBJ-NEXT:   Section (2) .rela.text {
// OBJ-NEXT:     0x0 R_AARCH64_ADD_ABS_LO12_NC some_label 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]
