; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @foo1(i32* align 32 %a) #0 {
entry:
  %0 = load i32* %a, align 4
  ret i32 %0

; CHECK-LABEL: @foo1
; CHECK-DAG: load i32* %a, align 32
; CHECK: ret i32
}

