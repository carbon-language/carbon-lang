; This function contains intervening instructions which should be moved out of the way
; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | not grep call

int %Test(int %X) {
entry:
        %tmp.1 = seteq int %X, 0
        br bool %tmp.1, label %then.0, label %endif.0

then.0:
        %tmp.4 = add int %X, 1
        ret int %tmp.4

endif.0:
        %tmp.10 = add int %X, -1
        %tmp.8 = call int %Test(int %tmp.10)
	%DUMMY = add int %X, 1                ;; This should not prevent elimination
        ret int %tmp.8
}

