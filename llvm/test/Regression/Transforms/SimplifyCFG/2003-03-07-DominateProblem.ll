; RUN: as < %s | opt -simplifycfg -disable-output

implementation   ; Functions:

void %test(int* %ldo, bool %c, bool %d) {
bb9:
	br bool %c, label %bb11, label %bb10

bb10:		; preds = %bb9
	br label %bb11

bb11:		; preds = %bb10, %bb9
	%reg330 = phi int* [ null, %bb10 ], [ %ldo, %bb9 ]
	br label %bb20

bb20:		; preds = %bb23, %bb25, %bb27, %bb11
	store int* %reg330, int** null
	br bool %d, label %bb20, label %done

done:
	ret void
}
