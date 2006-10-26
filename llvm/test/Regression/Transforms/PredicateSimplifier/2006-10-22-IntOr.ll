; RUN: llvm-as < %s | opt -predsimplify -instcombine -simplifycfg &&
; RUN: llvm-as < %s | opt -predsimplify -instcombine -simplifycfg | llvm-dis | grep -v declare | grep -c fail | grep 1 &&
; RUN: llvm-as < %s | opt -predsimplify -instcombine -simplifycfg | llvm-dis | grep -v declare | grep -c pass | grep 1

int %test1(int %x, int %y) {
entry:
	%tmp2 = or int %x, %y		; <int> [#uses=1]
	%tmp = seteq int %tmp2, 0		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %return

cond_true:		; preds = %entry
	%tmp4 = seteq int %x, 0		; <bool> [#uses=1]
	br bool %tmp4, label %cond_true5, label %return

cond_true5:		; preds = %cond_true
	%tmp6 = call int %fail( )		; <int> [#uses=0]
	ret int %tmp6

return:		; preds = %cond_next7
	ret int 0
}

int %test2(int %x, int %y) {
entry:
	%tmp2 = or int %x, %y		; <int> [#uses=1]
	%tmp = setne int %tmp2, 0		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %return

cond_true:		; preds = %entry
	%tmp4 = seteq int %x, 0		; <bool> [#uses=1]
	br bool %tmp4, label %cond_true5, label %return

cond_true5:		; preds = %cond_true
	%tmp6 = call int %pass( )		; <int> [#uses=0]
	ret int %tmp6

return:		; preds = %cond_next7
	ret int 0
}

declare int %fail()
declare int %pass()
