; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify -disable-output

void %gs_image_next() {
entry:
	%tmp = load uint* null		; <uint> [#uses=2]
	br bool false, label %cond_next21, label %UnifiedReturnBlock

cond_next21:		; preds = %entry
	br bool false, label %cond_next42, label %UnifiedReturnBlock

cond_next42:		; preds = %cond_next21
	br label %cond_true158

cond_next134:		; preds = %cond_true158
	%tmp1571 = seteq uint 0, %min		; <bool> [#uses=0]
	ret void

cond_true158:		; preds = %cond_true158, %cond_next42
	%tmp47 = sub uint %tmp, 0		; <uint> [#uses=2]
	%tmp49 = setle uint %tmp47, 0		; <bool> [#uses=1]
	%min = select bool %tmp49, uint %tmp47, uint 0		; <uint> [#uses=2]
	%tmp92 = add uint %min, 0		; <uint> [#uses=1]
	%tmp101 = seteq uint %tmp92, %tmp		; <bool> [#uses=1]
	br bool %tmp101, label %cond_next134, label %cond_true158

UnifiedReturnBlock:		; preds = %cond_next21, %entry
	ret void
}
