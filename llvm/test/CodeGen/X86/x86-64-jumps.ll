; RUN: llc < %s 
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"

define i8 @test1() nounwind ssp {
entry:
  %0 = select i1 undef, i8* blockaddress(@test1, %bb), i8* blockaddress(@test1, %bb6) ; <i8*> [#uses=1]
  indirectbr i8* %0, [label %bb, label %bb6]

bb:                                               ; preds = %entry
  ret i8 1

bb6:                                              ; preds = %entry
  ret i8 2
}

