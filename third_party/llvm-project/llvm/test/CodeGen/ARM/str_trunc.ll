; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define void @test1(i32 %v, i16* %ptr) {
        %tmp = trunc i32 %v to i16              ; <i16> [#uses=1]
        store i16 %tmp, i16* %ptr
        ret void
}

define void @test2(i32 %v, i8* %ptr) {
        %tmp = trunc i32 %v to i8               ; <i8> [#uses=1]
        store i8 %tmp, i8* %ptr
        ret void
}

; CHECK: strh
; CHECK-NOT: strh

; CHECK: strb
; CHECK-NOT: strb

