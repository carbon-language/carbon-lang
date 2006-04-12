; There should be exactly one vxor here.
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 --enable-unsafe-fp-math | grep vxor | wc -l | grep 1 &&

; There should be exactly one vsplti here.
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 --enable-unsafe-fp-math | grep vsplti | wc -l | grep 1


void %VXOR(<4 x float>* %P1, <4 x int>* %P2, <4 x float>* %P3) {
        %tmp = load <4 x float>* %P3
        %tmp3 = load <4 x float>* %P1
        %tmp4 = mul <4 x float> %tmp, %tmp3
        store <4 x float> %tmp4, <4 x float>* %P3
        store <4 x float> zeroinitializer, <4 x float>* %P1
        store <4 x int> zeroinitializer, <4 x int>* %P2
        ret void
}

void %VSPLTI(<4 x int>* %P2, <8 x short>* %P3) {
        store <4 x int> cast (<16 x sbyte> < sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1, sbyte -1 > to <4 x int>), <4 x int>* %P2
        store <8 x short> < short -1, short -1, short -1, short -1, short -1, short -1, short -1, short -1 >, <8 x short>* %P3
        ret void
}

