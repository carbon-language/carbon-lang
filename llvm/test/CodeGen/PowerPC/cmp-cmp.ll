; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | not grep mfcr

void %test(long %X) {
        %tmp1 = and long %X, 3          ; <long> [#uses=1]
        %tmp = setgt long %tmp1, 2              ; <bool> [#uses=1]
        br bool %tmp, label %UnifiedReturnBlock, label %cond_true

cond_true:              ; preds = %entry
        tail call void %test(long 0)
        ret void

UnifiedReturnBlock:             ; preds = %entry
        ret void
}

