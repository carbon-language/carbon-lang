; -simplifycfg is not folding blocks if there is a PHI node involved.  This 
; should be fixed eventually

; RUN: as < %s | opt -simplifycfg | dis | not grep br

int %main(int %argc) {
        br label %InlinedFunctionReturnNode

InlinedFunctionReturnNode:                                      ;[#uses=1]
        %X = phi int [ 7, %0 ]          ; <int> [#uses=1]
        %Y = add int %X, %argc          ; <int> [#uses=1]
        ret int %Y
}

