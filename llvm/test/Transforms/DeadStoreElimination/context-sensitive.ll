; RUN: opt < %s -basicaa -dse -S | not grep DEAD
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

declare void @ext()

define i32* @caller() {
        %P = malloc i32         ; <i32*> [#uses=4]
        %DEAD = load i32* %P            ; <i32> [#uses=1]
        %DEAD2 = add i32 %DEAD, 1               ; <i32> [#uses=1]
        store i32 %DEAD2, i32* %P
        call void @ext( )
        store i32 0, i32* %P
        ret i32* %P
}

