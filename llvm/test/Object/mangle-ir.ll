; RUN: llvm-as %s -o - | llvm-nm - | FileCheck %s

target datalayout = "m:o"

; CHECK: T _f
define void @f() {
  ret void
}
