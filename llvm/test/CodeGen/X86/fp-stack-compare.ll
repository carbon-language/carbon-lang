; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mcpu=i386 | \
; RUN:   grep {fucomi.*st.\[12\]}
; PR1012

float %foo(float *%col.2.0) {
        %tmp = load float* %col.2.0             ; <float> [#uses=3]
        %tmp16 = setlt float %tmp, 0.000000e+00         ; <bool> [#uses=1]
        %tmp20 = sub float -0.000000e+00, %tmp          ; <float> [#uses=1]
        %iftmp.2.0 = select bool %tmp16, float %tmp20, float %tmp
	ret float %iftmp.2.0
}

