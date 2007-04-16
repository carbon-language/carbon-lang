; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   opt -predsimplify -instcombine -simplifycfg | llvm-dis | \
; RUN:   grep -v declare | grep -c pass | grep 2

int %test(int %x, int %y) {
entry:
        %tmp2 = setlt int %x, %y
	%tmp = setne bool %tmp2, true
	br bool %tmp, label %cond_true, label %return

cond_true:		; preds = %entry
	%tmp4 = seteq int %x, %y		; <bool> [#uses=1]
	br bool %tmp4, label %cond_true5, label %cond_false

cond_true5:		; preds = %cond_true
	%tmp6 = call int %pass1( )		; <int> [#uses=1]
	ret int %tmp6

cond_false:
	%tmp8 = call int %pass2( )		; <int> [#uses=1]
	ret int %tmp8

return:		; preds = %cond_next7
	ret int 0
}

declare int %pass1()
declare int %pass2()
