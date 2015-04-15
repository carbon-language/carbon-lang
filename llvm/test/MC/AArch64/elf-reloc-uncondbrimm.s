// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck -check-prefix=OBJ %s

        b somewhere
        bl somewhere

// OBJ:      Relocations [
// OBJ-NEXT:   Section {{.*}} .rela.text {
// OBJ-NEXT:     0x0 R_AARCH64_JUMP26 somewhere 0x0
// OBJ-NEXT:     0x4 R_AARCH64_CALL26 somewhere 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]
