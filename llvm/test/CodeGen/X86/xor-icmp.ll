; RUN: llc < %s -march=x86    | FileCheck %s -check-prefix=X32
; RUN: llc < %s -march=x86-64 | FileCheck %s -check-prefix=X64

define i32 @t(i32 %a, i32 %b) nounwind ssp {
entry:
; X32:     t:
; X32:     xorb
; X32-NOT: andb
; X32-NOT: shrb
; X32:     testb $64
; X32:     jne

; X64:     t:
; X64-NOT: setne
; X64:     xorl
; X64:     testb $64
; X64:     jne
  %0 = and i32 %a, 16384
  %1 = icmp ne i32 %0, 0
  %2 = and i32 %b, 16384
  %3 = icmp ne i32 %2, 0
  %4 = xor i1 %1, %3
  br i1 %4, label %bb1, label %bb

bb:                                               ; preds = %entry
  %5 = tail call i32 (...)* @foo() nounwind       ; <i32> [#uses=1]
  ret i32 %5

bb1:                                              ; preds = %entry
  %6 = tail call i32 (...)* @bar() nounwind       ; <i32> [#uses=1]
  ret i32 %6
}

declare i32 @foo(...)

declare i32 @bar(...)
