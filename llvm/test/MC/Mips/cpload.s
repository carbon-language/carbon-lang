# RUN: llvm-mc %s -arch=mips -mcpu=mips32r2 | FileCheck %s -check-prefix=ASM
#
# RUN: llvm-mc %s -arch=mips -mcpu=mips32r2 -filetype=obj -o -| \
# RUN: llvm-objdump -d -r -arch=mips - | \
# RUN: FileCheck %s -check-prefix=OBJ

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -filetype=obj -o -| \
# RUN: llvm-objdump -d -r -arch=mips - | \
# RUN: FileCheck %s -check-prefix=OBJ64

# ASM:    .text
# ASM:    .option pic2
# ASM:    .set noreorder
# ASM:    .cpload $25
# ASM:    .set reorder

# OBJ:    .text
# OBJ:    lui $gp, 0
# OBJ: R_MIPS_HI16 _gp_disp
# OBJ:    addiu $gp, $gp, 0
# OBJ: R_MIPS_LO16 _gp_disp
# OBJ:    addu $gp, $gp, $25

# OBJ64: .text
# OBJ64-NOT: lui $gp, 0
# OBJ64-NOT: addiu $gp, $gp, 0
# OBJ64-NOT: addu $gp, $gp, $25

        .text
        .option pic2
        .set noreorder
        .cpload $25
        .set reorder
