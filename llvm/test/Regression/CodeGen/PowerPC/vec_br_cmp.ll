; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vcmpeqfp. &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep mfcr

; A predicate compare used immediately by a branch should not generate an mfcr.

void %test(<4 x float>* %A, <4 x float>* %B) {
        %tmp = load <4 x float>* %A
        %tmp3 = load <4 x float>* %B
        %tmp = tail call int %llvm.ppc.altivec.vcmpeqfp.p( int 1, <4 x float> %tmp, <4 x float> %tmp3 )
        %tmp = seteq int %tmp, 0
        br bool %tmp, label %cond_true, label %UnifiedReturnBlock

cond_true:
        store <4 x float> zeroinitializer, <4 x float>* %B
        ret void

UnifiedReturnBlock:
        ret void
}

declare int %llvm.ppc.altivec.vcmpeqfp.p(int, <4 x float>, <4 x float>)

