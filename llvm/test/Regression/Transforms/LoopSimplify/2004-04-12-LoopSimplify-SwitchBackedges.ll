; RUN: llvm-as < %s | opt -loopsimplify -disable-output

implementation   

void %test() {
loopentry.0:
	br label %loopentry.1

loopentry.1:
	%pixel.4 = phi int [ 0, %loopentry.0 ], [ %pixel.4, %loopentry.1], [ %tmp.370, %then.6 ], [ %tmp.370, %then.6 ]
	br bool false, label %then.6, label %loopentry.1

then.6:
	%tmp.370 = add int 0, 0		; <int> [#uses=2]
	switch uint 0, label %label.7 [
		 uint 6408, label %loopentry.1
		 uint 32841, label %loopentry.1
	]

label.7:		; preds = %then.6
	ret void
}
