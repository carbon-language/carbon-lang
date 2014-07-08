# Error checking for malformed abi related directives
# RUN: not llvm-mc -triple mips-unknown-unknown %s 2>&1 | FileCheck %s
# CHECK: .text
    .module fp=3
# CHECK      : mips-abi-bad.s:4:16: error: unsupported option
# CHECK-NEXT : .module fp=3
# CHECK-NEXT :           ^

    .set fp=xx,6
# CHECK      :mips-abi-bad.s:5:15: error: unexpected token in statement
# CHECK-NEXT :    .set fp=xx,6
# CHECK-NEXT :              ^

# CHECK       :.set mips16
    .set mips16
    .module fp=32

# CHECK      :mips-abi-bad.s:14:13: error: .module directive must come before any code
# CHECK-NEXT :    .module fp=32
# CHECK-NEXT :            ^
