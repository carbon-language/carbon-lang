; RUN: llc < %s -mcpu=atom -mtriple=i686-linux -march=x86-64 | FileCheck %s

; Additional tests for 64-bit divide bypass

define i64 @Test_get_quotient(i64 %a, i64 %b) nounwind {
; CHECK: Test_get_quotient:
; CHECK: orq %rsi, %rcx
; CHECK-NEXT: testq $-65536, %rcx
; CHECK-NEXT: je
; CHECK: idivq
; CHECK: ret
; CHECK: divw
; CHECK: ret
  %result = sdiv i64 %a, %b
  ret i64 %result
}

define i64 @Test_get_remainder(i64 %a, i64 %b) nounwind {
; CHECK: Test_get_remainder:
; CHECK: orq %rsi, %rcx
; CHECK-NEXT: testq $-65536, %rcx
; CHECK-NEXT: je
; CHECK: idivq
; CHECK: ret
; CHECK: divw
; CHECK: ret
  %result = srem i64 %a, %b
  ret i64 %result
}

define i64 @Test_get_quotient_and_remainder(i64 %a, i64 %b) nounwind {
; CHECK: Test_get_quotient_and_remainder:
; CHECK: orq %rsi, %rcx
; CHECK-NEXT: testq $-65536, %rcx
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
