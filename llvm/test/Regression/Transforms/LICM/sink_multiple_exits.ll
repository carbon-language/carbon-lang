; This testcase ensures that we can sink instructions from loops with
; multiple exits.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | grep -C1 mul | grep Out: 

int %test(int %N, bool %C) {
Entry:
	br label %Loop
Loop:
        %N_addr.0.pn = phi int [ %dec, %ContLoop ], [ %N, %Entry ]
        %tmp.6 = mul int %N, %N_addr.0.pn
        %tmp.7 = sub int %tmp.6, %N
        %dec = add int %N_addr.0.pn, -1
        br bool %C, label %ContLoop, label %Out1
ContLoop:
        %tmp.1 = setne int %N_addr.0.pn, 1
        br bool %tmp.1, label %Loop, label %Out2
Out1:
	ret int %tmp.7
Out2:
        ret int %tmp.7
}
