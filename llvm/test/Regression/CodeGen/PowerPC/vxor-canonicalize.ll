; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vxor | wc -l | grep 1
; There should be exactly one vxor here, not two.

void %test(<4 x float>* %P1, <4 x int>* %P2) {
        store <4 x float> zeroinitializer, <4 x float>* %P1
        store <4 x int> zeroinitializer, <4 x int>* %P2
        ret void
}

