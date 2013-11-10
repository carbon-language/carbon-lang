; RUN: opt < %s -simplifycfg -S | FileCheck %s

;CHECK: @foo
;CHECK: and i32 %c1, %k
;CHECK: icmp eq i32
;CHECK: and i32 %c2, %k
;CHECK: icmp eq i32
;CHECK: or i1
;CHECK: ret
define i32 @foo(i32 %k, i32 %c1, i32 %c2) {
  %1 = and i32 %c1, %k
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %8, label %3

; <label>:3                                       ; preds = %0
  %4 = and i32 %c2, %k
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %8, label %6

; <label>:6                                       ; preds = %3
  %7 = tail call i32 (...)* @bar() nounwind
  br label %8

; <label>:8                                       ; preds = %3, %0, %6
  ret i32 undef
}

declare i32 @bar(...)
