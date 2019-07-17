; RUN: llc %s -o - | FileCheck %s

target triple = "aarch64-unknown-unknown-eabi"

define void @test_tcommit() #0 {
  tail call void @llvm.aarch64.tcommit()
  ret void
}

declare void @llvm.aarch64.tcommit() #1

attributes #0 = { "target-features"="+tme" }
attributes #1 = { nounwind }

; CHECK-LABEL: test_tcommit
; CHECK: tcommit
