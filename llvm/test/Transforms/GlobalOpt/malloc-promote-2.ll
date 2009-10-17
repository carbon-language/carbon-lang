; RUN: opt < %s -globalopt -globaldce -S | not grep malloc
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

@G = internal global i32* null          ; <i32**> [#uses=3]

define void @init() {
        %P = malloc i32, i32 100                ; <i32*> [#uses=1]
        store i32* %P, i32** @G
        %GV = load i32** @G             ; <i32*> [#uses=1]
        %GVe = getelementptr i32* %GV, i32 40           ; <i32*> [#uses=1]
        store i32 20, i32* %GVe
        ret void
}

define i32 @get() {
        %GV = load i32** @G             ; <i32*> [#uses=1]
        %GVe = getelementptr i32* %GV, i32 40           ; <i32*> [#uses=1]
        %V = load i32* %GVe             ; <i32> [#uses=1]
        ret i32 %V
}

