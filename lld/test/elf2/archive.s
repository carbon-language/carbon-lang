# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %S/Inputs/archive.s -o %t2
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %S/Inputs/archive2.s -o %t3
# RUN: llvm-ar rcs %tar %t2 %t3
# RUN: lld -flavor gnu2 %t %tar -o %tout
# RUN: llvm-nm %tout | FileCheck %s
# REQUIRES: x86

# Nothing here. Just needed for the linker to create a undefined _start symbol.

.quad end

.weak foo
.quad foo

# CHECK: T _start
# CHECK: T end
# CHECK: w foo
