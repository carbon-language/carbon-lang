; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 --enable-unsafe-fp-math | grep vxor | wc -l | grep 1
; There should be exactly one vxor here.

void %test(<4 x float>* %P1, <4 x int>* %P2, <4 x float>* %P3) {
        %tmp = load <4 x float>* %P3
        %tmp3 = load <4 x float>* %P1
        %tmp4 = mul <4 x float> %tmp, %tmp3
        store <4 x float> %tmp4, <4 x float>* %P3
        store <4 x float> zeroinitializer, <4 x float>* %P1
        store <4 x int> zeroinitializer, <4 x int>* %P2
        ret void
}

