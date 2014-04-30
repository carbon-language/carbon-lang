// RUN: llvm-mc -filetype=obj -triple arm64_be %s | llvm-readobj -section-data -sections | FileCheck %s

// CHECK: 0000: 00123456 789ABCDE
foo:    .xword 0x123456789abcde
