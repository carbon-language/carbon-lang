; If the result of an instruction is only used outside of the loop, sink
; the instruction to the exit blocks instead of executing it on every
; iteration of the loop.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | grep -C1 mul | grep Out: 

int %test(int %N) {
Entry:
	br label %Loop
Loop:
        %N_addr.0.pn = phi int [ %dec, %Loop ], [ %N, %Entry ]
        %tmp.6 = mul int %N, %N_addr.0.pn
        %dec = add int %N_addr.0.pn, -1
        %tmp.1 = setne int %N_addr.0.pn, 1
        br bool %tmp.1, label %Loop, label %Out
Out:
	ret int %tmp.6
}
