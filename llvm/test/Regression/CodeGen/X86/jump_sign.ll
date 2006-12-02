; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep jns
int %f(int %X) {
entry:
        %tmp1 = add int %X, 1           ; <int> [#uses=1]
        %tmp = setlt int %tmp1, 0               ; <bool> [#uses=1]
        br bool %tmp, label %cond_true, label %cond_next

cond_true:              ; preds = %entry
        %tmp2 = tail call int (...)* %bar( )            ; <int> [#uses=0]
        br label %cond_next

cond_next:              ; preds = %entry, %cond_true
        %tmp3 = tail call int (...)* %baz( )            ; <int> [#uses=0]
        ret int undef
}

declare int %bar(...)

declare int %baz(...)

