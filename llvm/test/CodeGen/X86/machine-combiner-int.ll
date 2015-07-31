; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=x86-64 < %s | FileCheck %s

; Verify that integer multiplies are reassociated. The first multiply in 
; each test should be independent of the result of the preceding add (lea).

define i16 @reassociate_muls_i16(i16 %x0, i16 %x1, i16 %x2, i16 %x3) {
; CHECK-LABEL: reassociate_muls_i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    leal   (%rdi,%rsi), %eax
; CHECK-NEXT:    imull  %ecx, %edx
; CHECK-NEXT:    imull  %edx, %eax
; CHECK-NEXT:    retq
  %t0 = add i16 %x0, %x1
  %t1 = mul i16 %x2, %t0
  %t2 = mul i16 %x3, %t1
  ret i16 %t2
}

define i32 @reassociate_muls_i32(i32 %x0, i32 %x1, i32 %x2, i32 %x3) {
; CHECK-LABEL: reassociate_muls_i32:
; CHECK:       # BB#0:
; CHECK-NEXT:    leal   (%rdi,%rsi), %eax
; CHECK-NEXT:    imull  %ecx, %edx
; CHECK-NEXT:    imull  %edx, %eax
; CHECK-NEXT:    retq
  %t0 = add i32 %x0, %x1
  %t1 = mul i32 %x2, %t0
  %t2 = mul i32 %x3, %t1
  ret i32 %t2
}

define i64 @reassociate_muls_i64(i64 %x0, i64 %x1, i64 %x2, i64 %x3) {
; CHECK-LABEL: reassociate_muls_i64:
; CHECK:       # BB#0:
; CHECK-NEXT:    leaq   (%rdi,%rsi), %rax
; CHECK-NEXT:    imulq  %rcx, %rdx
; CHECK-NEXT:    imulq  %rdx, %rax
; CHECK-NEXT:    retq
  %t0 = add i64 %x0, %x1
  %t1 = mul i64 %x2, %t0
  %t2 = mul i64 %x3, %t1
  ret i64 %t2
}
