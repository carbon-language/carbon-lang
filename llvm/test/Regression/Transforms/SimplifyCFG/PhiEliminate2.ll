; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep br

int %test(bool %C, int %V1, int %V2) {
entry:
        br bool %C, label %then, label %Cont

then:
        %V3 = or int %V2, %V1
        br label %Cont
Cont:
	%V4 = phi int [%V1, %entry], [%V3, %then]
	call int %test(bool false, int 0, int 0)           ;; don't fold into preds
        ret int %V1
}

