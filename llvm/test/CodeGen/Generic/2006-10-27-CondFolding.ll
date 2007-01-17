; RUN: llvm-upgrade < %s | llvm-as | llc 

void %start_pass_huff(int %gather_statistics) {
entry:
        %tmp = seteq int %gather_statistics, 0          ; <bool> [#uses=1]
        br bool false, label %cond_next22, label %bb166

cond_next22:            ; preds = %entry
        %bothcond = and bool false, %tmp                ; <bool> [#uses=1]
        br bool %bothcond, label %bb34, label %bb46

bb34:           ; preds = %cond_next22
        ret void

bb46:           ; preds = %cond_next22
        ret void

bb166:          ; preds = %entry
        ret void
}

