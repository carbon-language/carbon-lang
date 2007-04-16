; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | not grep IMPLICIT_DEF

void %foo(long %X) {
entry:
        %tmp1 = and long %X, 3          ; <long> [#uses=1]
        %tmp = setgt long %tmp1, 2              ; <bool> [#uses=1]
        br bool %tmp, label %UnifiedReturnBlock, label %cond_true

cond_true:              ; preds = %entry
        %tmp = tail call int (...)* %bar( )             ; <int> [#uses=0]
        ret void

UnifiedReturnBlock:             ; preds = %entry
        ret void
}

declare int %bar(...)

