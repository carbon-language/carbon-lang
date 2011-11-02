; RUN: opt < %s -indvars -S -enable-iv-rewrite=true | FileCheck %s
;
; PR11279: Assertion !IVLimit->getType()->isPointerTy()
;
; Test a non-integer BECount. It doesn't make sense, but that's what
; falls out of SCEV. Since it's an i8*, we never adjust in a way that
; would convert it to an integer type.
;
; enable-iv-rewrite=false does not currently perform LFTR when the the
; taken count is a pointer expression, but that will change son.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin"

; CHECK: @test8
; CHECK: loop:
; CHECK: icmp ne
define i8 @test8(i8* %buf, i8* %end) nounwind {
  br label %loopguard

loopguard:
  %guard = icmp ult i8* %buf, %end
  br i1 %guard, label %preheader, label %exit

preheader:
  br label %loop

loop:
  %p.01.us.us = phi i8* [ %buf, %preheader ], [ %gep, %loop ]
  %s = phi i8 [0, %preheader], [%snext, %loop]
  %gep = getelementptr inbounds i8* %p.01.us.us, i64 1
  %snext = load i8* %gep
  %cmp = icmp ult i8* %gep, %end
  br i1 %cmp, label %loop, label %exit

exit:
  ret i8 %snext
}
