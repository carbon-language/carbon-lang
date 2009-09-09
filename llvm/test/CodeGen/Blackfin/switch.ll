; RUN: llc < %s -march=bfin -verify-machineinstrs > %t

define i32 @foo(i32 %A, i32 %B, i32 %C) {
entry:
	switch i32 %A, label %out [
		i32 1, label %bb
		i32 0, label %bb13
	]

bb:		; preds = %entry
	ret i32 1

bb13:		; preds = %entry
	ret i32 1

out:		; preds = %entry
	ret i32 0
}
