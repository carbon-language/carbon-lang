; RUN: llc -march=ppc64le -mcpu=pwr8 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Indirect calls requires a full stub creation
define void @test_indirect(void ()* nocapture %fp) {
; CHECK-LABEL: @test_indirect
  tail call void %fp()
; CHECK-DAG: std 2, 24(1)
; CHECK-DAG: mr 12, 3
; CHECK-DAG: mtctr 3
; CHECK: bctrl
; CHECK-NEXT: ld 2, 24(1)
  ret void
}

