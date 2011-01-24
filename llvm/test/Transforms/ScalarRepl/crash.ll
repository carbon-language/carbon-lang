; RUN: opt -scalarrepl %s -disable-output
; RUN: opt -scalarrepl-ssa %s -disable-output

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; PR9017
define void @test1() nounwind readnone ssp {
entry:
  %l_72 = alloca i32*, align 8
  unreachable

for.cond:                                         ; preds = %for.cond
  %tmp1.i = load i32** %l_72, align 8
  store i32* %tmp1.i, i32** %l_72, align 8
  br label %for.cond

if.end:                                           ; No predecessors!
  ret void
}
