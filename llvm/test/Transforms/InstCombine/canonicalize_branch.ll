; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %X, i32 %Y) {
        %C = icmp ne i32 %X, %Y
        br i1 %C, label %T, label %F

; CHECK: @test1
; CHECK: %C = icmp eq i32 %X, %Y
; CHECK: br i1 %C, label %F, label %T

T:
        ret i32 12
F:
        ret i32 123
}

define i32 @test2(i32 %X, i32 %Y) {
        %C = icmp ule i32 %X, %Y
        br i1 %C, label %T, label %F

; CHECK: @test2
; CHECK: %C = icmp ugt i32 %X, %Y
; CHECK: br i1 %C, label %F, label %T

T:
        ret i32 12
F:
        ret i32 123
}

define i32 @test3(i32 %X, i32 %Y) {
        %C = icmp uge i32 %X, %Y
        br i1 %C, label %T, label %F

; CHECK: @test3
; CHECK: %C = icmp ult i32 %X, %Y
; CHECK: br i1 %C, label %F, label %T

T:
        ret i32 12
F:
        ret i32 123
}

