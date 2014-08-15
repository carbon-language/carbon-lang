// RUN: llvm-mc -triple=x86_64-apple-darwin %s | FileCheck %s

a:
b:
        .uleb128 a-b

// CHECK:        .uleb128 a-b
