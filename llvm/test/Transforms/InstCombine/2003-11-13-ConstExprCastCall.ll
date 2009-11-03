; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

declare void @free(i8*)

define void @test(i32* %X) {
        call void (...)* bitcast (void (i8*)* @free to void (...)*)( i32* %X )          ; <i32>:1 [#uses=0]
; CHECK: %tmp = bitcast i32* %X to i8*
; CHECK: call void @free(i8* %tmp)
        ret void
; CHECK: ret void
}
