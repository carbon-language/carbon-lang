// RUN: %clang -### --analyze %s 2>&1 | FileCheck %s
// CHECK: -D__clang_analyzer__
