; RUN: llc -mtriple=aarch64-unknown-unknown -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mattr=-perfmon -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=NOPERFMON

define i64 @test_readcyclecounter() nounwind {
  ; CHECK-LABEL:   test_readcyclecounter:
  ; PERFMON-NEXT:   mrs x0, PMCCNTR_EL0
  ; NOPERFMON-NEXT: mov x0, xzr
  ; CHECK-NEXT:     ret
  %tmp0 = call i64 @llvm.readcyclecounter()
  ret i64 %tmp0
}

declare i64 @llvm.readcyclecounter()
