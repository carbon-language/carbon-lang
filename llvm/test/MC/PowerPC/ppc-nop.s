# RUN: llvm-mc -filetype=obj -triple=powerpc-unknown-linux-gnu %s | llvm-readobj -s -sd - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux-gnu %s | llvm-readobj -s -sd - | FileCheck %s

blr
.p2align 3
blr

.byte 0x42
.p2align 2

# CHECK:  0000: 4E800020 60000000 4E800020 42000000

