# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=i586 < %s 2>&1 | FileCheck %s

# CHECK: error: unable to find instruction-level scheduling information for target triple 'x86_64-unknown-unknown' and cpu 'i586'.
