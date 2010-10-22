; RUN: llc < %s -march=arm -mattr=+v6 | FileCheck %s

define i32 @test1(i32 %X) {
; CHECK: test1
; CHECK: rev16 r0, r0
        %tmp1 = lshr i32 %X, 8
        %X15 = bitcast i32 %X to i32
        %tmp4 = shl i32 %X15, 8
        %tmp2 = and i32 %tmp1, 16711680
        %tmp5 = and i32 %tmp4, -16777216
        %tmp9 = and i32 %tmp1, 255
        %tmp13 = and i32 %tmp4, 65280
        %tmp6 = or i32 %tmp5, %tmp2
        %tmp10 = or i32 %tmp6, %tmp13
        %tmp14 = or i32 %tmp10, %tmp9
        ret i32 %tmp14
}

define i32 @test2(i32 %X) {
; CHECK: test2
; CHECK: revsh r0, r0
        %tmp1 = lshr i32 %X, 8
        %tmp1.upgrd.1 = trunc i32 %tmp1 to i16
        %tmp3 = trunc i32 %X to i16
        %tmp2 = and i16 %tmp1.upgrd.1, 255
        %tmp4 = shl i16 %tmp3, 8
        %tmp5 = or i16 %tmp2, %tmp4
        %tmp5.upgrd.2 = sext i16 %tmp5 to i32
        ret i32 %tmp5.upgrd.2
}
