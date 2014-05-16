; RUN: llc < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

@m = internal global i32 0, align 4
@n = internal global i32 0, align 4

define void @f1(i32 %a1, i32 %a2) {
; CHECK-LABEL: f1:
; CHECK: adrp x{{[0-9]+}}, _MergedGlobals
; CHECK-NOT: adrp
  store i32 %a1, i32* @m, align 4
  store i32 %a2, i32* @n, align 4
  ret void
}

; CHECK:        .local _MergedGlobals
; CHECK:        .comm  _MergedGlobals,8,8

