; RUN: opt -S -mergefunc < %s | FileCheck %s
%Si = type <{ i32 }>

define void @sum(%Si* noalias sret %a, i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = add i32 %sum2, %y
  ret void
}

define void @add(%Si* noalias sret %a, i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = add i32 %sum2, %y
  ret void
}

; Make sure we transfer the parameter attributes to the call site.

; CHECK-LABEL: define void @sum(%Si* noalias sret, i32, i32)
; CHECK: tail call void @add(%Si* noalias sret %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; CHECK: ret void
