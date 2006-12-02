; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify -disable-output

fastcc void %_ov_splice(int %n1, int %n2, int %ch2) {
entry:
	%tmp = setgt int %n1, %n2		; <bool> [#uses=1]
	%n.0 = select bool %tmp, int %n2, int %n1		; <int> [#uses=1]
	%tmp104 = setlt int 0, %ch2		; <bool> [#uses=1]
	br bool %tmp104, label %cond_true105, label %return

cond_true95:		; preds = %cond_true105
	ret void

bb98:		; preds = %cond_true105
	ret void

cond_true105:		; preds = %entry
	%tmp94 = setgt int %n.0, 0		; <bool> [#uses=1]
	br bool %tmp94, label %cond_true95, label %bb98

return:		; preds = %entry
	ret void
}
