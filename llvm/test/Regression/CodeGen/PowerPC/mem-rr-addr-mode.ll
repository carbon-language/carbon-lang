; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep 'li.*16' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep addi

; Codegen lvx (R+16) as t = li 16,  lvx t,R
; This shares the 16 between the two loads.

void %func(<4 x float>* %a, <4 x float>* %b) {
	%tmp1 = getelementptr <4 x float>* %b, int 1		
	%tmp = load <4 x float>* %tmp1		
	%tmp3 = getelementptr <4 x float>* %a, int 1		
	%tmp4 = load <4 x float>* %tmp3		
	%tmp5 = mul <4 x float> %tmp, %tmp4		
	%tmp8 = load <4 x float>* %b		
	%tmp9 = add <4 x float> %tmp5, %tmp8		
	store <4 x float> %tmp9, <4 x float>* %a
	ret void
}
