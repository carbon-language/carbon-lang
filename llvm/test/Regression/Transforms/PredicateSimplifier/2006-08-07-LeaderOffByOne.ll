; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify -disable-output

void %safe_strcpy(uint %size1) {
entry:
	%tmp = seteq uint %size1, 0		; <bool> [#uses=1]
	br bool %tmp, label %return, label %strlen.exit

strlen.exit:		; preds = %entry
	%tmp = cast ulong 0 to uint		; <uint> [#uses=2]
	%tmp6 = setlt uint %tmp, %size1		; <bool> [#uses=1]
	br bool %tmp6, label %cond_true7, label %cond_false19

cond_true7:		; preds = %strlen.exit
	%tmp9 = seteq uint %tmp, 0		; <bool> [#uses=1]
	br bool %tmp9, label %cond_next15, label %cond_true10

cond_true10:		; preds = %cond_true7
	ret void

cond_next15:		; preds = %cond_true7
	ret void

cond_false19:		; preds = %strlen.exit
	ret void

return:		; preds = %entry
	ret void
}
