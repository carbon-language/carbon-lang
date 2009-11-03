; A store or load cannot alias a global if the accessed amount is larger then
; the global.

; RUN: opt < %s -basicaa -gvn -instcombine -S | not grep load
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@B = global i16 8               ; <i16*> [#uses=2]

define i16 @test(i32* %P) {
        %X = load i16* @B               ; <i16> [#uses=1]
        store i32 7, i32* %P
        %Y = load i16* @B               ; <i16> [#uses=1]
        %Z = sub i16 %Y, %X             ; <i16> [#uses=1]
        ret i16 %Z
}

