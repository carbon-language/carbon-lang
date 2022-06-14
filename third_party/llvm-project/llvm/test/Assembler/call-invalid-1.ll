; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare void @f()

define void @g() {
  call void @f() align 8
; CHECK: error: call instructions may not have an alignment
  ret void
}
