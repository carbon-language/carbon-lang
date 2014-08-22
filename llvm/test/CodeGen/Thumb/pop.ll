; RUN: llc < %s -mtriple=thumb-apple-darwin | FileCheck %s
; rdar://7268481

define void @t(i8* %a, ...) nounwind {
; CHECK-LABEL:      t:
; CHECK:      pop {r3}
; CHECK-NEXT: add sp, #12
; CHECK-NEXT: bx r3
entry:
  %a.addr = alloca i8, i32 4
  call void @llvm.va_start(i8* %a.addr)
  ret void
}

declare void @llvm.va_start(i8*) nounwind
