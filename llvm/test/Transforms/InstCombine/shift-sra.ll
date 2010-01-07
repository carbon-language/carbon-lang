; RUN: opt < %s -instcombine -S | FileCheck %s


define i32 @test1(i32 %X, i8 %A) {
        %shift.upgrd.1 = zext i8 %A to i32              ; <i32> [#uses=1]
        ; can be logical shift.
        %Y = ashr i32 %X, %shift.upgrd.1                ; <i32> [#uses=1]
        %Z = and i32 %Y, 1              ; <i32> [#uses=1]
        ret i32 %Z
; CHECK: @test1
; CHECK: lshr i32 %X, %shift.upgrd.1 
}

define i32 @test2(i8 %tmp) {
        %tmp3 = zext i8 %tmp to i32             ; <i32> [#uses=1]
        %tmp4 = add i32 %tmp3, 7                ; <i32> [#uses=1]
        %tmp5 = ashr i32 %tmp4, 3               ; <i32> [#uses=1]
        ret i32 %tmp5
; CHECK: @test2
; CHECK: lshr i32 %tmp4, 3
}
