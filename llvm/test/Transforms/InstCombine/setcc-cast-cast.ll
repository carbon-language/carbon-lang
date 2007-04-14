; This test case was reduced from MultiSource/Applications/hbd. It makes sure
; that folding doesn't happen in case a zext is applied where a sext should have
; been when a setcc is used with two casts.
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    not grep {br bool false}
; END.

int %bug(ubyte %inbuff) {
entry:
	%tmp = bitcast ubyte %inbuff to sbyte   ; <sbyte> [#uses=1]
	%tmp = sext sbyte %tmp to int		; <int> [#uses=3]
	%tmp = seteq int %tmp, 1		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	br label %bb

cond_next:		; preds = %entry
	%tmp3 = seteq int %tmp, -1		; <bool> [#uses=1]
	br bool %tmp3, label %cond_true4, label %cond_next5

cond_true4:		; preds = %cond_next
	br label %bb

cond_next5:		; preds = %cond_next
	%tmp7 = setgt int %tmp, 1		; <bool> [#uses=1]
	br bool %tmp7, label %cond_true8, label %cond_false

cond_true8:		; preds = %cond_next5
	br label %cond_next9

cond_false:		; preds = %cond_next5
	br label %cond_next9

cond_next9:		; preds = %cond_false, %cond_true8
	%iftmp.1.0 = phi int [ 42, %cond_true8 ], [ 23, %cond_false ]		; <int> [#uses=1]
	br label %return

bb:		; preds = %cond_true4, %cond_true
	br label %return

return:		; preds = %bb, %cond_next9
	%retval.0 = phi int [ 17, %bb ], [ %iftmp.1.0, %cond_next9 ]		; <int> [#uses=1]
	ret int %retval.0
}
