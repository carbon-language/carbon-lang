; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-p:32:32"

define i32* @test(i32* %P) {
        %V = ptrtoint i32* %P to i32            ; <i32> [#uses=1]
        %P2 = inttoptr i32 %V to i32*           ; <i32*> [#uses=1]
        ret i32* %P2
; CHECK: ret i32* %P
}

