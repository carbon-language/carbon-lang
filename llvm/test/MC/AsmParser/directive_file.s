# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

        .file "hello"
        .file 1 "world"

# CHECK: .file "hello"
# CHECK: .file 1 "world"

