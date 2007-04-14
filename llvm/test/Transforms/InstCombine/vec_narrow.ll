; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep {add float}

%V = type <4 x float>

float %test(%V %A, %V %B, float %f) {
        %C = insertelement %V %A, float %f, uint 0
        %D = add %V %C, %B
        %E = extractelement %V %D, uint 0
        ret float %E
}

