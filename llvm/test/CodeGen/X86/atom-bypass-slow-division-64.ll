; RUN: llc < %s -mcpu=atom -march=x86-64 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; Additional tests for 64-bit divide bypass

define i64 @Test_get_quotient(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: Test_get_quotient:
; CHECK: movq %rdi, %rax
; CHECK: orq %rsi, %rax
; CHECK-NEXT: testq $-65536, %rax
; CHECK-NEXT: je
; CHECK: idivq
; CHECK: ret
; CHECK: divw
; CHECK: ret
  %result = sdiv i64 %a, %b
  ret i64 %result
}

define i64 @Test_get_remainder(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: Test_get_remainder:
; CHECK: movq %rdi, %rax
; CHECK: orq %rsi, %rax
; CHECK-NEXT: testq $-65536, %rax
; CHECK-NEXT: je
; CHECK: idivq
; CHECK: ret
; CHECK: divw
; CHECK: ret
  %result = srem i64 %a, %b
  ret i64 %result
}

define i64 @Test_get_quotient_and_remainder(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: Test_get_quotient_and_remainder:
; CHECK: movq %rdi, %rax
; CHECK: orq %rsi, %rax
; CHECK-NEXT: testq $-65536, %rax
; CHECK-NEXT: je
; CHECK: idivq
; CHECK: divw
; CHECK: addq
; CHECK: ret
; CHECK-NOT: idivq
; CHECK-NOT: divw
  %resultdiv = sdiv i64 %a, %b
  %resultrem = srem i64 %a, %b
  %result = add i64 %resultdiv, %resultrem
  ret i64 %result
}
