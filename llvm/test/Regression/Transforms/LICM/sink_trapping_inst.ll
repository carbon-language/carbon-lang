; Potentially trapping instructions may be sunk as long as they are guaranteed
; to be executed.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | grep -C1 div | grep Out: 

int %test(int %N) {
Entry:
	br label %Loop
Loop:
        %N_addr.0.pn = phi int [ %dec, %Loop ], [ %N, %Entry ]
        %tmp.6 = div int %N, %N_addr.0.pn
        %dec = add int %N_addr.0.pn, -1
        %tmp.1 = setne int %N_addr.0.pn, 0
        br bool %tmp.1, label %Loop, label %Out
Out:
	ret int %tmp.6
}
