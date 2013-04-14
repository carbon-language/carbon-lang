; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;CHECK: @foo
;CHECK: store <4 x i32>
;CHECK: ret
define i32 @foo(i32* nocapture %A, i32 %n) {
entry:
  %call = tail call i32 (...)* @bar() #2
  %mul = mul nsw i32 %n, 5
  %add = add nsw i32 %mul, 9
  store i32 %add, i32* %A, align 4
  %mul1 = mul nsw i32 %n, 9
  %add2 = add nsw i32 %mul1, 9
  %arrayidx3 = getelementptr inbounds i32* %A, i64 1
  store i32 %add2, i32* %arrayidx3, align 4
  %mul4 = shl i32 %n, 3
  %add5 = add nsw i32 %mul4, 9
  %arrayidx6 = getelementptr inbounds i32* %A, i64 2
  store i32 %add5, i32* %arrayidx6, align 4
  %mul7 = mul nsw i32 %n, 10
  %add8 = add nsw i32 %mul7, 9
  %arrayidx9 = getelementptr inbounds i32* %A, i64 3
  store i32 %add8, i32* %arrayidx9, align 4
  ret i32 undef
}

  ; We can still vectorize the stores below.

declare i32 @bar(...)
