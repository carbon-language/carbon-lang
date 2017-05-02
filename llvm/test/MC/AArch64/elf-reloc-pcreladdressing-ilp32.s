// RUN: llvm-mc -target-abi=ilp32 -triple=aarch64-none-linux-gnu \
// RUN:   -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck -check-prefix=OBJ-ILP32 %s
        adr x2, some_label
        adrp x5, some_label

        adrp x5, :got:some_label
        ldr w0, [x5, #:got_lo12:some_label]

// OBJ-ILP32:      Relocations [
// OBJ-ILP32-NEXT:   Section {{.*}} .rela.text {
// OBJ-ILP32-NEXT:     0x0 R_AARCH64_P32_ADR_PREL_LO21    some_label 0x0
// OBJ-ILP32-NEXT:     0x4 R_AARCH64_P32_ADR_PREL_PG_HI21 some_label 0x0
// OBJ-ILP32-NEXT:     0x8 R_AARCH64_P32_ADR_GOT_PAGE     some_label 0x0
// OBJ-ILP32-NEXT:     0xC R_AARCH64_P32_LD32_GOT_LO12_NC some_label 0x0
// OBJ-ILP32-NEXT:   }
// OBJ-ILP32-NEXT: ]
