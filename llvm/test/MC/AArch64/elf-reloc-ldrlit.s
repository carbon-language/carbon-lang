// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck -check-prefix=OBJ %s

        ldr x0, some_label
        ldr w3, some_label
        ldrsw x9, some_label
        prfm pldl3keep, some_label

// OBJ:      Relocations [
// OBJ-NEXT:   Section (1) .text {
// OBJ-NEXT:     0x0 R_AARCH64_LD_PREL_LO19 some_label 0x0
// OBJ-NEXT:     0x4 R_AARCH64_LD_PREL_LO19 some_label 0x0
// OBJ-NEXT:     0x8 R_AARCH64_LD_PREL_LO19 some_label 0x0
// OBJ-NEXT:     0xC R_AARCH64_LD_PREL_LO19 some_label 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]
