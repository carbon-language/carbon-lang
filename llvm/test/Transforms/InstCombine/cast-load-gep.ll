; RUN: opt < %s -instcombine -globaldce -S | \
; RUN:   not grep Array
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; Pulling the cast out of the load allows us to eliminate the load, and then 
; the whole array.

        %op = type { float }
        %unop = type { i32 }
@Array = internal constant [1 x %op* (%op*)*] [ %op* (%op*)* @foo ]             ; <[1 x %op* (%op*)*]*> [#uses=1]

define %op* @foo(%op* %X) {
        ret %op* %X
}

define %unop* @caller(%op* %O) {
        %tmp = load %unop* (%op*)** bitcast ([1 x %op* (%op*)*]* @Array to %unop* (%op*)**); <%unop* (%op*)*> [#uses=1]
        %tmp.2 = call %unop* %tmp( %op* %O )            ; <%unop*> [#uses=1]
        ret %unop* %tmp.2
}

