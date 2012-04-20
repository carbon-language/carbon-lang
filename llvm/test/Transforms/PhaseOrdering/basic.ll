; RUN: opt -O3 -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"

declare i8* @malloc(i64)
declare void @free(i8*)


; PR2338
define void @test1() nounwind ssp {
  %retval = alloca i32, align 4
  %i = alloca i8*, align 8
  %call = call i8* @malloc(i64 1)
  store i8* %call, i8** %i, align 8
  %tmp = load i8** %i, align 8
  store i8 1, i8* %tmp
  %tmp1 = load i8** %i, align 8
  call void @free(i8* %tmp1)
  ret void

; CHECK: @test1
; CHECK-NEXT: ret void
}
