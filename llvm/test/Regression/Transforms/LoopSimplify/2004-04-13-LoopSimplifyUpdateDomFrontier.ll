; RUN: llvm-as < %s | opt -scalarrepl -loopsimplify -licm -disable-output

implementation   ; Functions:

void %inflate() {
entry:
	br label %loopentry.0.outer1111

loopentry.0.outer1111:		; preds = %entry, %loopentry.0.outer1111, %label.11, %then.41
	%left.0.ph1107 = phi uint [ %tmp.1172, %then.41 ], [ 0, %entry ], [ %tmp.1172, %label.11 ], [ %left.0.ph1107, %loopentry.0.outer1111 ]		; <uint> [#uses=2]
	%tmp.1172 = sub uint %left.0.ph1107, 0		; <uint> [#uses=2]
	switch uint 0, label %label.11 [
		 uint 23, label %loopentry.0.outer1111
		 uint 13, label %then.41
	]

label.11:		; preds = %loopentry.0.outer1111
	br label %loopentry.0.outer1111

then.41:		; preds = %loopentry.0.outer1111
	br label %loopentry.0.outer1111
}
