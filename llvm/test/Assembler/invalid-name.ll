; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: expected function name
define void @"zed\00bar"() {
  ret void
}
