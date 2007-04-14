; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    grep {%bothcond =}
bool %Doit_bb(int %i.0) {
bb:             ; preds = %newFuncRoot
        %tmp = setgt int %i.0, 0             ; <bool> [#uses=1]
        %tmp.not = xor bool %tmp, true          ; <bool> [#uses=1]
        %tmp2 = setgt int %i.0, 8            ; <bool> [#uses=1]
        %bothcond = or bool %tmp.not, %tmp2             ; <bool> [#uses=1]
        br bool %bothcond, label %exitTrue, label %exitFalse

exitTrue:             ; preds = %bb
        ret bool true

exitFalse:            ; preds = %bb
        ret bool false

}
