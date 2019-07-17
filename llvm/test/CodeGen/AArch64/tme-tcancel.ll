; RUN: llc %s -o - | FileCheck %s

target triple = "aarch64-unknown-unknown-eabi"

define void @test_tcancel() #0 {
  tail call void @llvm.aarch64.tcancel(i64 0) #1
  unreachable
}

declare void @llvm.aarch64.tcancel(i64 immarg) #1

attributes #0 = { "target-features"="+tme" }
attributes #1 = { nounwind noreturn }

; CHECK-LABEL: test_tcancel
; CHECK: tcancel
