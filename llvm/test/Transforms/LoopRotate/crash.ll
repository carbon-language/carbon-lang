; RUN: opt -loop-rotate %s -disable-output

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; PR8955 - Rotating an outer loop that has a condbr for a latch block.
define void @test1() nounwind ssp {
entry:
  br label %lbl_283

lbl_283:                                          ; preds = %if.end, %entry
  br i1 undef, label %if.else, label %if.then

if.then:                                          ; preds = %lbl_283
  br i1 undef, label %if.end, label %for.condthread-pre-split

for.condthread-pre-split:                         ; preds = %if.then
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %for.condthread-pre-split
  br i1 undef, label %lbl_281, label %for.cond

lbl_281:                                          ; preds = %if.end, %for.cond
  br label %if.end

if.end:                                           ; preds = %lbl_281, %if.then
  br i1 undef, label %lbl_283, label %lbl_281

if.else:                                          ; preds = %lbl_283
  ret void
}
