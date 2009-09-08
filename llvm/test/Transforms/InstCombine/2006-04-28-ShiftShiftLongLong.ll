; RUN: opt %s -instcombine | llvm-dis | grep shl
; RUN: opt %s -instcombine | llvm-dis | notcast

; This cannot be turned into a sign extending cast!

define i64 @test(i64 %X) {
        %Y = shl i64 %X, 16             ; <i64> [#uses=1]
        %Z = ashr i64 %Y, 16            ; <i64> [#uses=1]
        ret i64 %Z
}

