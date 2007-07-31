; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin | %prcontext {pinsrw \$2} 1 | grep "movl \$1"
; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin | not grep movss

@G = global <4 x float> zeroinitializer

define void @test(i32 *%P1, i32* %P2, float *%FP) {
        %T = load float* %FP
        store i32 0, i32* %P1

        %U = load <4 x float>* @G
        store i32 1, i32* %P1
        %V = insertelement <4 x float> %U, float %T, i32 1
        store <4 x float> %V, <4 x float>* @G

        ret void
}
