; To reduce register pressure, if a load is hoistable out of the loop, and the
; result of the load is only used outside of the loop, sink the load instead of
; hoisting it!
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | grep -C1 load | grep Out: 

%X = global int 5

int %test(int %N) {
Entry:
	br label %Loop
Loop:
        %N_addr.0.pn = phi int [ %dec, %Loop ], [ %N, %Entry ]
        %tmp.6 = load int* %X
        %dec = add int %N_addr.0.pn, -1
        %tmp.1 = setne int %N_addr.0.pn, 1
        br bool %tmp.1, label %Loop, label %Out
Out:
	ret int %tmp.6
}
