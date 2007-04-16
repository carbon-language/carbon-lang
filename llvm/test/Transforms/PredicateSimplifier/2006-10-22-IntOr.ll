; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   opt -predsimplify -instcombine -simplifycfg  | llvm-dis > %t
; RUN: grep -v declare %t | not grep fail
; RUN: grep -v declare %t | grep -c pass | grep 3

int %test1(int %x, int %y) {
entry:
	%tmp2 = or int %x, %y		; <int> [#uses=1]
	%tmp = seteq int %tmp2, 0		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %return

cond_true:		; preds = %entry
	%tmp4 = seteq int %x, 0		; <bool> [#uses=1]
	br bool %tmp4, label %cond_true5, label %cond_false

cond_true5:		; preds = %cond_true
	%tmp6 = call int %pass( )		; <int> [#uses=1]
	ret int %tmp6

cond_false:
	%tmp8 = call int %fail ( )		; <int> [#uses=1]
	ret int %tmp8

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

declare int %fail()
declare int %pass()
declare int %pass1()
declare int %pass2()
