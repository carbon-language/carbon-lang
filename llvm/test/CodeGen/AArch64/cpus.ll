; This tests that llc accepts all valid AArch64 CPUs


; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=generic 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=carmel 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a35 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a34 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a53 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a55 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a57 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a65 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a65ae 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a72 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a73 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a75 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a76ae 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a76 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a77 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a78 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-x1 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=neoverse-e1 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=neoverse-n1 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=neoverse-n2 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=neoverse-v1 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=exynos-m3 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=exynos-m4 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=exynos-m5 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=falkor 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=saphira 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=kryo 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=thunderx2t99 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=thunderx3t110 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=tsv110 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=apple-latest 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=a64fx 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=invalidcpu 2>&1 | FileCheck %s --check-prefix=INVALID

; CHECK-NOT: {{.*}}  is not a recognized processor for this target
; INVALID: {{.*}}  is not a recognized processor for this target

define i32 @f(i64 %z) {
	ret i32 0
}
