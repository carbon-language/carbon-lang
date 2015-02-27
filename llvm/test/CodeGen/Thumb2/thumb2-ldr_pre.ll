; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32* @test1(i32* %X, i32* %dest) {
        %Y = getelementptr i32, i32* %X, i32 4               ; <i32*> [#uses=2]
        %A = load i32* %Y               ; <i32> [#uses=1]
        store i32 %A, i32* %dest
        ret i32* %Y
}

; CHECK: ldr{{.*}}!

define i32 @test2(i32 %a, i32 %b) {
        %tmp1 = sub i32 %a, 64          ; <i32> [#uses=2]
        %tmp2 = inttoptr i32 %tmp1 to i32*              ; <i32*> [#uses=1]
        %tmp3 = load i32* %tmp2         ; <i32> [#uses=1]
        %tmp4 = sub i32 %tmp1, %b               ; <i32> [#uses=1]
        %tmp5 = add i32 %tmp4, %tmp3            ; <i32> [#uses=1]
        ret i32 %tmp5
}

; CHECK: ldr{{.*}}!

define i8* @test3(i8* %X, i32* %dest) {
        %tmp1 = getelementptr i8, i8* %X, i32 4
        %tmp2 = load i8* %tmp1
        %tmp3 = sext i8 %tmp2 to i32
        store i32 %tmp3, i32* %dest
        ret i8* %tmp1
}

; CHECK: ldrsb{{.*}}!

