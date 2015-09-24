# RUN: llvm-mc -filetype=obj -triple aarch64 %s -o -| llvm-readobj -h | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnu %s -o -| llvm-readobj -h | FileCheck %s

# CHECK: OS/ABI: SystemV (0x0)
