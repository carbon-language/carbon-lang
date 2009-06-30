; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {ldr\\W*pc,} | count 1

define i32 @foo(i32 %a) nounwind  {
entry:
	switch i32 %a, label %bb4 [
		 i32 1, label %bb5
		 i32 2, label %bb1
		 i32 3, label %bb2
		 i32 5, label %bb3
	]

bb1:		; preds = %entry
	ret i32 1

bb2:		; preds = %entry
	ret i32 1234

bb3:		; preds = %entry
	ret i32 3456

bb4:		; preds = %entry
	ret i32 0

bb5:		; preds = %entry
	ret i32 12
}
