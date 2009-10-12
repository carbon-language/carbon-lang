; RUN: opt < %s -instcombine -S | FileCheck %s

define i16 @test1(i16 %a) {
        %tmp = zext i16 %a to i32               ; <i32> [#uses=2]
        %tmp21 = lshr i32 %tmp, 8               ; <i32> [#uses=1]
; CHECK: %tmp21 = lshr i16 %a, 8
        %tmp5 = mul i32 %tmp, 5         ; <i32> [#uses=1]
; CHECK: %tmp5 = mul i16 %a, 5
        %tmp.upgrd.32 = or i32 %tmp21, %tmp5            ; <i32> [#uses=1]
; CHECK: %tmp.upgrd.32 = or i16 %tmp21, %tmp5
        %tmp.upgrd.3 = trunc i32 %tmp.upgrd.32 to i16           ; <i16> [#uses=1]
        ret i16 %tmp.upgrd.3
; CHECK: ret i16 %tmp.upgrd.32
}

