; RUN: not llvm-as < %s -o /dev/null -f
; PR1042

int %foo() {
        %A = invoke int %foo( )
                        to label %L unwind label %L             ; <int> [#uses=1]

L:              ; preds = %0, %0
        ret int %A
}
