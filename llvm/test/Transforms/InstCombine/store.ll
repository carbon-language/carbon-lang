; RUN: opt < %s -instcombine -S | FileCheck %s

define void @test1(i32* %P) {
        store i32 undef, i32* %P
        store i32 123, i32* undef
        store i32 124, i32* null
        ret void
; CHECK: @test1(
; CHECK-NEXT: store i32 undef, i32* null
; CHECK-NEXT: ret void
}

define void @test2(i32* %P) {
        %X = load i32* %P               ; <i32> [#uses=1]
        %Y = add i32 %X, 0              ; <i32> [#uses=1]
        store i32 %Y, i32* %P
        ret void
; CHECK: @test2
; CHECK-NEXT: ret void
}

