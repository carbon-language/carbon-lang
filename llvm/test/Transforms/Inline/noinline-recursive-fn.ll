; The inliner should never inline recursive functions into other functions.
; This effectively is just peeling off the first iteration of a loop, and the
; inliner heuristics are not set up for this.

; RUN: opt -inline %s -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.3"

@g = common global i32 0                          ; <i32*> [#uses=1]

define internal void @foo(i32 %x) nounwind ssp {
entry:
  %0 = icmp slt i32 %x, 0                         ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb

bb:                                               ; preds = %entry
  %1 = sub nsw i32 %x, 1                          ; <i32> [#uses=1]
  call void @foo(i32 %1) nounwind ssp
  store volatile i32 1, i32* @g, align 4
  ret void

return:                                           ; preds = %entry
  ret void
}


;; CHECK: @bonk
;; CHECK: call void @foo(i32 42)
define void @bonk() nounwind ssp {
entry:
  call void @foo(i32 42) nounwind ssp
  ret void
}



;; Here is an indirect case that should not be infinitely inlined.

define internal void @f1(i32 %x, i8* %Foo, i8* %Bar) nounwind ssp {
entry:
  %0 = bitcast i8* %Bar to void (i32, i8*, i8*)*
  %1 = sub nsw i32 %x, 1
  call void %0(i32 %1, i8* %Foo, i8* %Bar) nounwind
  store volatile i32 42, i32* @g, align 4
  ret void
}

define internal void @f2(i32 %x, i8* %Foo, i8* %Bar) nounwind ssp {
entry:
  %0 = icmp slt i32 %x, 0                         ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb

bb:                                               ; preds = %entry
  %1 = bitcast i8* %Foo to void (i32, i8*, i8*)*  ; <void (i32, i8*, i8*)*> [#uses=1]
  call void %1(i32 %x, i8* %Foo, i8* %Bar) nounwind
  store volatile i32 13, i32* @g, align 4
  ret void

return:                                           ; preds = %entry
  ret void
}


; CHECK: @top_level
; CHECK: call void @f2(i32 122
; Here we inline one instance of the cycle, but we don't want to completely
; unroll it.
define void @top_level() nounwind ssp {
entry:
  call void @f2(i32 123, i8* bitcast (void (i32, i8*, i8*)* @f1 to i8*), i8* bitcast (void (i32, i8*, i8*)* @f2 to i8*)) nounwind ssp
  ret void
}
