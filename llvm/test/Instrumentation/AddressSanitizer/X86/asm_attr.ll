; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+sse2 -asm-instrumentation=address -asan-instrument-assembly | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: mov_no_attr
; CHECK-NOT: callq __sanitizer_sanitize_load8@PLT
; CHECK-NOT: callq __sanitizer_sanitize_store8@PLT
define void @mov_no_attr(i64* %dst, i64* %src) {
  tail call void asm sideeffect "movq ($1), %rax  \0A\09movq %rax, ($0)  \0A\09", "r,r,~{memory},~{rax},~{dirflag},~{fpsr},~{flags}"(i64* %dst, i64* %src)
  ret void
}

; CHECK-LABEL: mov_sanitize
; CHECK: callq __sanitizer_sanitize_load8@PLT
; CHECK: callq __sanitizer_sanitize_store8@PLT
define void @mov_sanitize(i64* %dst, i64* %src) sanitize_address {
  tail call void asm sideeffect "movq ($1), %rax  \0A\09movq %rax, ($0)  \0A\09", "r,r,~{memory},~{rax},~{dirflag},~{fpsr},~{flags}"(i64* %dst, i64* %src)
  ret void
}
