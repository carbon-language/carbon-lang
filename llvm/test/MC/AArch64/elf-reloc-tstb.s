// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck -check-prefix=OBJ %s

        tbz x6, #45, somewhere
        tbnz w3, #15, somewhere

// OBJ:      Relocations [
// OBJ-NEXT:   Section {{.*}} .rela.text {
// OBJ-NEXT:     0x0  R_AARCH64_TSTBR14 somewhere 0x0
// OBJ-NEXT:     0x4  R_AARCH64_TSTBR14 somewhere 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]
