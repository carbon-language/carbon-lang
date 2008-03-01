; This testcase shows a bug where an common subexpression exists, but there
; is no shared dominator block that the expression can be hoisted out to.
;
; RUN: llvm-as < %s | opt -load-vn -gcse | llvm-dis | not grep load

define i32 @test(i32* %P) {
        store i32 5, i32* %P
        %Z = load i32* %P               ; <i32> [#uses=1]
        ret i32 %Z
}

