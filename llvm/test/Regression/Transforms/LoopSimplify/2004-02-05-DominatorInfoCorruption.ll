; RUN: llvm-as < %s | opt -loopsimplify -verify -licm -disable-output
implementation   ; Functions:

void %.subst_48() {
entry:
	br label %loopentry.0

loopentry.0:            ; preds = %entry, %loopentry.0
        br bool false, label %loopentry.0, label %loopentry.2

loopentry.2:            ; preds = %loopentry.0, %loopentry.2
        %tmp.968 = setle int 0, 3               ; <bool> [#uses=1]
        br bool %tmp.968, label %loopentry.2, label %UnifiedReturnBlock

UnifiedReturnBlock:             ; preds = %entry, %loopentry.2
        ret void
}
