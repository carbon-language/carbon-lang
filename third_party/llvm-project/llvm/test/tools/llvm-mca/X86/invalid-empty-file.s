# RUN: not llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=btver2 -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: no assembly instructions found.
