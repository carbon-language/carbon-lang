; RUN: opt < %s -indmemrem -S | grep bounce | grep noalias
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

declare i8* @malloc(i32)

@g = external global i8*

define void @test() {
  %A = bitcast i8* (i32) * @malloc to i8*
  store i8* %A, i8** @g
  ret void
}
