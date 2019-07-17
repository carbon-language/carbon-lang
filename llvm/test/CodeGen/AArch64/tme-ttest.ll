; RUN: llc %s -o - | FileCheck %s

target triple = "aarch64-unknown-unknown-eabi"

define i64 @test_ttest() #0 {
  %r = tail call i64 @llvm.aarch64.ttest()
  ret i64 %r
}

declare i64 @llvm.aarch64.ttest() #1

attributes #0 = { "target-features"="+tme" }
attributes #1 = { nounwind }

; CHECK-LABEL: test_ttest
; CHECK: ttest x
