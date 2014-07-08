# RUN: not llvm-mc %s -triple mips-unknown-unknown -mcpu=mips64r2 2>&1 | FileCheck %s
# CHECK: .text



        .set fp=xx
# CHECK     : error: 'set fp=xx'option requires O32 ABI
# CHECK     : .set fp=xx
# CHECK     :          ^
