; RUN: llc -mcpu=nehalem -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @use(<2 x double>)

; CHECK-LABEL: @test
; CHECK: callq round

; Function Attrs: nounwind uwtable
define void @test() {
entry:
  %tmp = call <2 x double> @llvm.round.v2f64(<2 x double> undef)
  call void @use(<2 x double> %tmp)
  ret void
}

; Function Attrs: nounwind readonly
declare <2 x double> @llvm.round.v2f64(<2 x double>) #0

attributes #0 = { nounwind readonly }

