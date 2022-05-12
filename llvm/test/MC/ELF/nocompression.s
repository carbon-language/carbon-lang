// REQUIRES: !zlib
// RUN: not llvm-mc -filetype=obj -compress-debug-sections=zlib -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s
// RUN: not llvm-mc -filetype=obj -compress-debug-sections=zlib-gnu -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

// CHECK: llvm-mc{{[^:]*}}: error: build tools with zlib to enable -compress-debug-sections
