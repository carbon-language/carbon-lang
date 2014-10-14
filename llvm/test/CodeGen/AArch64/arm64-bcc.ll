; RUN: llc < %s -mtriple=arm64-apple-darwint  | FileCheck %s
; Checks for conditional branch b.vs

; Function Attrs: nounwind
define i32 @add(i32, i32) {
entry:
  %2 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %0, i32 %1)
  %3 = extractvalue { i32, i1 } %2, 1
  br i1 %3, label %6, label %4

; <label>:4                                       ; preds = %entry
  %5 = extractvalue { i32, i1 } %2, 0
  ret i32 %5

; <label>:6                                       ; preds = %entry
  tail call void @llvm.trap()
  unreachable
; CHECK: b.vs
}

; Function Attrs: nounwind readnone
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

; Function Attrs: noreturn nounwind
declare void @llvm.trap()

