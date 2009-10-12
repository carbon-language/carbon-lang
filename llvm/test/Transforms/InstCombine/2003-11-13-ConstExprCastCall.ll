; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @free(i8*)

define void @test(i32* %X) {
        call void (...)* bitcast (void (i8*)* @free to void (...)*)( i32* %X )          ; <i32>:1 [#uses=0]
; CHECK: %tmp = bitcast i32* %X to i8*
; CHECK: call void @free(i8* %tmp)
        ret void
; CHECK: ret void
}
