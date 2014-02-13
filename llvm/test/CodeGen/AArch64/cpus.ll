; This tests that llc accepts all valid AArch64 CPUs

; RUN: llc < %s -march=aarch64 -mcpu=generic 2>&1 | FileCheck %s
; RUN: llc < %s -march=aarch64 -mcpu=cortex-a53 2>&1 | FileCheck %s
; RUN: llc < %s -march=aarch64 -mcpu=cortex-a57 2>&1 | FileCheck %s
; RUN: llc < %s -march=aarch64 -mcpu=invalidcpu 2>&1 | FileCheck %s --check-prefix=INVALID

; CHECK-NOT: {{.*}}  is not a recognized processor for this target
; INVALID: {{.*}}  is not a recognized processor for this target

define i32 @f(i64 %z) {
	ret i32 0
}
