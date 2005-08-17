; RUN: llvm-as < %s | opt -loop-reduce -disable-output

int %image_to_texture(uint %indvar454) {
loopentry.1.outer:
	%j.2.1.ph = cast uint %indvar454 to int		; <int> [#uses=1]
	br label %loopentry.1

loopentry.1:		; preds = %label.5, %loopentry.1.outer
	%i.3 = phi int [ 0, %loopentry.1.outer ], [ %i.3.be, %loopentry.1 ]
	%tmp.390 = load int* null		; <int> [#uses=1]
	%tmp.392 = mul int %tmp.390, %j.2.1.ph		; <int> [#uses=1]
	%tmp.394 = add int %tmp.392, %i.3		; <int> [#uses=1]
	%i.3.be = add int %i.3, 1		; <int> [#uses=1]
	br bool false, label %loopentry.1, label %label.6

label.6:		; preds = %no_exit.1
	ret int %tmp.394
}
