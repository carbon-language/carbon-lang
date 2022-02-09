; RUN: llc -mtriple=arm-darwin < %s | FileCheck %s
; RUN: llc -O0 -mtriple=arm-darwin < %s | FileCheck %s

define void @foo(i8*, ...) {
  ret void
}
; CHECK-LABEL: {{^_?}}foo:
; CHECK-NOT: str
; CHECK: {{bx lr|mov pc, lr}}
declare void @llvm.va_start(i8*) nounwind
