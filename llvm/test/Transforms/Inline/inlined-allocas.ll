; RUN: opt -basicaa -dse -inline -S %s | FileCheck %s

declare void @external(i32* byval)
declare i32 @identity(i32* byval)

; An alloca in the inlinee should not force the tail to be stripped

define void @inlinee_with_alloca() {
  %local = alloca i32
  store i32 42, i32* %local, align 4
  tail call void @external(i32* byval %local)
  ret void
}

define void @inliner_without_alloca() {
  tail call void @inlinee_with_alloca()
  ret void
}

; CHECK-LABEL: inliner_without_alloca
; CHECK-NEXT: %local.i = alloca i32
; CHECK: tail call void @external
; CHECK: ret

; An alloca in the inliner should not force the tail to be stripped

define i32 @inliner_with_alloca() {
  %local = alloca i32
  store i32 42, i32* %local, align 4
  %1 = tail call i32 @identity(i32* byval %local)
  ret i32 %1
}

; CHECK-LABEL: inliner_with_alloca
; CHECK: %local = alloca i32
; CHECK: %1 = tail call i32 @identity
; CHECK: ret i32 %1

; Force the synthesis of the value through the byval parameter.
; The alloca should force the tail to be stripped

define void @inlinee_with_passthru(i32* byval %value) {
  tail call void @external(i32* byval %value)
  ret void
}

define void @strip_tail(i32* %value) {
  tail call void @inlinee_with_passthru(i32* %value)
  ret void
}

; CHECK-LABEL: strip_tail
; CHECK: %value1 = alloca i32
; CHECK-NOT: tail call void @external
; CHECK: ret void

