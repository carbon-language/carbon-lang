; The inliner should never inline recursive functions into other functions.
; This effectively is just peeling off the first iteration of a loop, and the
; inliner heuristics are not set up for this.

; RUN: opt -inline %s -S | grep "call void @foo(i32 42)"

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.3"

@g = common global i32 0                          ; <i32*> [#uses=1]

define internal void @foo(i32 %x) nounwind ssp {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = icmp slt i32 %x, 0                         ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb

bb:                                               ; preds = %entry
  %1 = sub nsw i32 %x, 1                          ; <i32> [#uses=1]
  call void @foo(i32 %1) nounwind ssp
  volatile store i32 1, i32* @g, align 4
  ret void

return:                                           ; preds = %entry
  ret void
}

define void @bonk() nounwind ssp {
entry:
  call void @foo(i32 42) nounwind ssp
  ret void
}
