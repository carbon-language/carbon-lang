# RUN: llvm-mc -filetype=obj -triple aarch64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=AARCH64-OSABI %s
# AARCH64-OSABI: OS/ABI: SystemV (0x0)

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnu %s -o -| llvm-readobj -h | FileCheck --check-prefix=AARCH64-LINUX-OSABI %s
# AARCH64-LINUX-OSABI: OS/ABI: GNU/Linux (0x3)
