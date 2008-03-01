; RUN: llvm-as < %s | opt -ipconstprop -instcombine | \
; RUN:    llvm-dis | grep {ret i1 true}
define internal i32 @foo(i1 %C) {
        br i1 %C, label %T, label %F

T:              ; preds = %0
        ret i32 52

F:              ; preds = %0
        ret i32 52
}

define i1 @caller(i1 %C) {
        %X = call i32 @foo( i1 %C )             ; <i32> [#uses=1]
        %Y = icmp ne i32 %X, 0          ; <i1> [#uses=1]
        ret i1 %Y
}

