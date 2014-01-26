# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 %s -o -| llvm-readobj -h | FileCheck %s

# This *MUST* match the output of gas compiled with the same triple.
# CHECK: Flags [ (0x50001006)

.abicalls
