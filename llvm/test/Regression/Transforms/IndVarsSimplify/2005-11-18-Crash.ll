; RUN: llvm-as < %s | opt -indvars -disable-output

%fixtab = external global [29 x [29 x [2 x uint]]]		; <[29 x [29 x [2 x uint]]]*> [#uses=1]

implementation   ; Functions:

void %init_optabs() {
entry:
	br label %no_exit.0

no_exit.0:		; preds = %no_exit.0, %entry
	%p.0.0 = phi uint* [ getelementptr ([29 x [29 x [2 x uint]]]* %fixtab, int 0, int 0, int 0, int 0), %entry ], [ %inc.0, %no_exit.0 ]		; <uint*> [#uses=1]
	%inc.0 = getelementptr uint* %p.0.0, int 1		; <uint*> [#uses=1]
	br bool false, label %no_exit.0, label %no_exit.1

no_exit.1:		; preds = %no_exit.0
	ret void
}
