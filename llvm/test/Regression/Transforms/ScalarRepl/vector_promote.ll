
; RUN: llvm-as < %s | opt -scalarrepl -disable-output &&
; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | not grep alloca

void %test(<4 x float>* %F, float %f) {
entry:
        %G = alloca <4 x float>, align 16               ; <<4 x float>*> [#uses=3]
        %tmp = load <4 x float>* %F             ; <<4 x float>> [#uses=2]
        %tmp3 = add <4 x float> %tmp, %tmp              ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp3, <4 x float>* %G
        %G = getelementptr <4 x float>* %G, int 0, int 0                ; <float*> [#uses=1]
        store float %f, float* %G
        %tmp4 = load <4 x float>* %G            ; <<4 x float>> [#uses=2]
        %tmp6 = add <4 x float> %tmp4, %tmp4            ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp6, <4 x float>* %F
        ret void
}

void %test2(<4 x float>* %F, float %f) {
entry:
        %G = alloca <4 x float>, align 16               ; <<4 x float>*> [#uses=3]
        %tmp = load <4 x float>* %F             ; <<4 x float>> [#uses=2]
        %tmp3 = add <4 x float> %tmp, %tmp              ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp3, <4 x float>* %G
        %tmp = getelementptr <4 x float>* %G, int 0, int 2              ; <float*> [#uses=1]
        store float %f, float* %tmp
        %tmp4 = load <4 x float>* %G            ; <<4 x float>> [#uses=2]
        %tmp6 = add <4 x float> %tmp4, %tmp4            ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp6, <4 x float>* %F
        ret void
}

void %test3(<4 x float>* %F, float* %f) {
entry:
        %G = alloca <4 x float>, align 16               ; <<4 x float>*> [#uses=2]
        %tmp = load <4 x float>* %F             ; <<4 x float>> [#uses=2]
        %tmp3 = add <4 x float> %tmp, %tmp              ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp3, <4 x float>* %G
        %tmp = getelementptr <4 x float>* %G, int 0, int 2              ; <float*> [#uses=1]
        %tmp = load float* %tmp         ; <float> [#uses=1]
        store float %tmp, float* %f
        ret void
}

void %test4(<4 x float>* %F, float* %f) {
entry:
        %G = alloca <4 x float>, align 16               ; <<4 x float>*> [#uses=2]
        %tmp = load <4 x float>* %F             ; <<4 x float>> [#uses=2]
        %tmp3 = add <4 x float> %tmp, %tmp              ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp3, <4 x float>* %G
        %G = getelementptr <4 x float>* %G, int 0, int 0                ; <float*> [#uses=1]
        %tmp = load float* %G           ; <float> [#uses=1]
        store float %tmp, float* %f
        ret void
}

