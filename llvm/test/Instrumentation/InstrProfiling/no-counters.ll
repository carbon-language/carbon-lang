;; No instrumentation should be emitted if there are no counter increments.

; RUN: opt < %s -instrprof -S | FileCheck %s
; CHECK-NOT: @__prf_cn
; CHECK-NOT: @__prf_dt
; CHECK-NOT: @__llvm_profile_runtime

define void @foo() {
  ret void
}
