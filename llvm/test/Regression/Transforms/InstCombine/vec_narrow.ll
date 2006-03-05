; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep 'add float'

%V = type <4 x float>

float %test(%V %A, %V %B, float %f) {
        %C = insertelement %V %A, float %f, uint 0
        %D = add %V %C, %B
        %E = extractelement %V %D, uint 0
        ret float %E
}

