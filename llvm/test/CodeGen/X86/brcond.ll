; RUN: llc < %s -march=x86 | FileCheck %s
; rdar://7475489

define i32 @t(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: t:
; CHECK: xorb
; CHECK-NOT: andb
; CHECK-NOT: shrb
; CHECK: testb $64
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
