// RUN: clang-tidy %s -- --serialize-diagnostics %t | FileCheck %s
// CHECK: :[[@LINE+1]]:12: error: expected ';' after struct [clang-diagnostic-error]
struct A {}
