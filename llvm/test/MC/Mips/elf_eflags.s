# These *MUST* match the output of gas compiled with the same triple and
# corresponding options (-mcpu=mips32 -> -mips32 for example).

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64r2 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64R2 %s
# MIPSEL-MIPS64R2: Flags [ (0x80001100)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64 %s
# MIPSEL-MIPS64: Flags [ (0x60001100)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32r2 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32R2 %s
# MIPSEL-MIPS32R2: Flags [ (0x70001000)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32 %s
# MIPSEL-MIPS32: Flags [ (0x50001000)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2 %s
# MIPS64EL-MIPS64R2: Flags [ (0x80000020)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64 %s
# MIPS64EL-MIPS64: Flags [ (0x60000020)
