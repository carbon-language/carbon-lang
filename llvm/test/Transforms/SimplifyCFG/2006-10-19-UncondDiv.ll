; PR957
; RUN: llvm-upgrade < %s | llvm-as | opt -simplifycfg | llvm-dis | \
; RUN:   not grep select

uint %test(uint %tmp) {
cond_false179:          ; preds = %cond_true
        %tmp181 = seteq uint %tmp, 0            ; <bool> [#uses=1]
        br bool %tmp181, label %cond_true182, label %cond_next185

cond_true182:           ; preds = %cond_false179
        br label %cond_next185

cond_next185:           ; preds = %cond_true182, %cond_false179
        %d0.3 = phi uint [ div (uint 1, uint 0), %cond_true182 ], [ %tmp,
%cond_false179 ]                ; <uint> [#uses=7]

        ret uint %d0.3
}

uint %test2(uint %tmp) {
cond_false179:          ; preds = %cond_true
        %tmp181 = seteq uint %tmp, 0            ; <bool> [#uses=1]
        br bool %tmp181, label %cond_true182, label %cond_next185

cond_true182:           ; preds = %cond_false179
        br label %cond_next185

cond_next185:           ; preds = %cond_true182, %cond_false179
        %d0.3 = phi uint [ div (uint 1, uint 0), %cond_true182 ], [ %tmp,
%cond_false179 ]                ; <uint> [#uses=7]
	call uint %test(uint 4)
        ret uint %d0.3
}

