# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 %s -o - | \
# RUN:   llvm-readobj -h | \
# RUN:     FileCheck %s -check-prefix=CHECK-OBJ
# RUN: llvm-mc -triple mipsel-unknown-linux -mcpu=mips32 %s -o -| \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM

# This *MUST* match the output of gas compiled with the same triple.
# CHECK-OBJ: Flags [ (0x50001400)

# CHECK-ASM: .nan 2008

.nan 2008
