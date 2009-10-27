; RUN: opt < %s -inline -S | FileCheck %s

define i32 @test1f(i32 %i) {
        ret i32 %i
}

define i32 @test1(i32 %W) {
        %X = call i32 @test1f(i32 7)
        %Y = add i32 %X, %W
        ret i32 %Y
; CHECK: @test1(
; CHECK-NEXT: %Y = add i32 7, %W
; CHECK-NEXT: ret i32 %Y
}

