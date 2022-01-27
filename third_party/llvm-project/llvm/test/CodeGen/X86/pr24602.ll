; RUN: llc < %s -mtriple=x86_64-unknown-unknown  | FileCheck %s

; PR24602: Make sure we don't barf on non-foldable code (with opaque constants).

; CHECK-LABEL: pr24602:
; CHECK-NEXT: # %bb.0
; CHECK-NEXT: movabsq $-10000000000, [[CST:%[a-z0-9]+]]
; CHECK-NEXT: imulq [[CST]], %rsi
; CHECK-NEXT: leaq (%rdi,%rsi,8), %rax
; CHECK-NEXT: movq [[CST]], (%rdi,%rsi,8)
; CHECK-NEXT: retq
define i64* @pr24602(i64* %p, i64 %n) nounwind {
  %mul = mul nsw i64 %n, -10000000000
  %add.ptr = getelementptr inbounds i64, i64* %p, i64 %mul
  store i64 -10000000000, i64* %add.ptr
  ret i64* %add.ptr
}
