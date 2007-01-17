; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify -disable-output

void %diff(int %N) {
entry:
	%tmp = setgt int %N, 0		; <bool> [#uses=1]
	br bool %tmp, label %bb519, label %bb744

bb519:		; preds = %entry
	%tmp720101 = setlt int %N, 0		; <bool> [#uses=1]
	br bool %tmp720101, label %bb744, label %bb744

bb744:		; preds = %bb519, %entry
	ret void
}
