; -simplifycfg is not folding blocks if there is a PHI node involved.  This 
; should be fixed eventually

; RUN: if as < %s | opt -simplifycfg | dis | grep br
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int %main(int %argc) {
        br label %InlinedFunctionReturnNode

InlinedFunctionReturnNode:                                      ;[#uses=1]
        %X = phi int [ 7, %0 ]          ; <int> [#uses=1]
        %Y = add int %X, %argc          ; <int> [#uses=1]
        ret int %Y
}

