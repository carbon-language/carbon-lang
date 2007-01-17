; RUN: not llvm-as < %s -o /dev/null -f
; PR1042

int %foo() {
        br bool false, label %L1, label %L2
L1:
        %A = invoke int %foo() to label %L unwind label %L

L2:
        br label %L
L:
        ret int %A
}
