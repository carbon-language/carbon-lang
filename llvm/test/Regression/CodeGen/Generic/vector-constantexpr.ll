; RUN: llvm-as < %s | llc
	
void ""(float* %inregs, float* %outregs) {
        %a_addr.i = alloca <4 x float>          ; <<4 x float>*> [#uses=1]
        store <4 x float> < float extractelement (<4 x float> undef, uint 3), float extractelement (<4 x float> undef, uint 0), float extractelement (<4 x float> undef, uint 1), float extractelement (<4 x float> undef, uint 2) >, <4 x float>* %a_addr.i
        ret void
}



