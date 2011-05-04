; RUN: llc < %s -march=ppc32 -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

declare void @bar(i64 %x, i64 %y)

; CHECK: li 4, 2
; CHECK: li {{[53]}}, 0
; CHECK: li 6, 3
; CHECK: mr {{[53]}}, {{[53]}}

define void @foo() {
  call void @bar(i64 2, i64 3)
  ret void
}
