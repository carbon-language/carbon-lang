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

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -mattr=n32 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-N32 %s
# MIPS64EL-MIPS64R2-N32: Flags [ (0x80000020)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 -mattr=n32 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64-N32 %s
# MIPS64EL-MIPS64-N32: Flags [ (0x60000020)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -mattr=n64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-N64 %s
# MIPS64EL-MIPS64R2-N64: Flags [ (0x80000000)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 %s -mattr=n64 -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64-N64 %s
# MIPS64EL-MIPS64-N64: Flags [ (0x60000000)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -mattr=o32 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-O32 %s
# MIPS64EL-MIPS64R2-O32: Flags [ (0x80001100)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 %s -mattr=o32 -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64-O32 %s
# MIPS64EL-MIPS64-O32: Flags [ (0x60001100)
	
# Default ABI for MIPS64 is N64 as opposed to GCC/GAS (N32)
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2 %s
# MIPS64EL-MIPS64R2: Flags [ (0x80000000)

# Default ABI for MIPS64 is N64 as opposed to GCC/GAS (N32)
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64 %s
# MIPS64EL-MIPS64: Flags [ (0x60000000)
