; RUN: opt -memdep -gvn -disable-output < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0"

define i32 @test2() nounwind ssp {
entry:
    ret i32 0

unreachable_block:
    %a = add i32 %a, 1
    ret i32 %a
}

