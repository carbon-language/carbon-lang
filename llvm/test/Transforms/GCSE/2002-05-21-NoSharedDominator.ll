; This testcase shows a bug where an common subexpression exists, but there
; is no shared dominator block that the expression can be hoisted out to.
;
; RUN: llvm-as < %s | opt -gcse | llvm-dis

define i32 @test(i32 %X, i32 %Y) {
        %Z = add i32 %X, %Y             ; <i32> [#uses=1]
        ret i32 %Z

Unreachable:            ; No predecessors!
        %Q = add i32 %X, %Y             ; <i32> [#uses=1]
        ret i32 %Q
}

