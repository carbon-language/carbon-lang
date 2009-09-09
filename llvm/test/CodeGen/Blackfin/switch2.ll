; RUN: llc < %s -march=bfin -verify-machineinstrs > %t

define i8* @FindChar(i8* %CurPtr) {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%tmp = load i8* null		; <i8> [#uses=1]
	switch i8 %tmp, label %bb [
		i8 0, label %bb7
		i8 120, label %bb7
	]

bb7:		; preds = %bb, %bb
	ret i8* null
}
