; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

define void @foo(i8*, ...) {
  ret void
}
; CHECK-LABEL: {{^_?}}foo:
; CHECK-NOT: movq
; CHECK: retq
