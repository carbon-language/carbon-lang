; RUN: llvm-as < %s | opt -condprop | llvm-dis | not grep phi

define i32 @foo(i1, i32, i32) {
prologue:
	br i1 %0, label %eq, label %ne

eq:		; preds = %prologue
	store i32 0, i32* inttoptr (i32 10000 to i32*)
	%3 = icmp eq i32 %1, %2		; <i1> [#uses=1]
	br label %join

ne:		; preds = %prologue
	%4 = icmp ne i32 %1, %2		; <i1> [#uses=1]
	br label %join

join:		; preds = %ne, %eq
	%5 = phi i1 [ %3, %eq ], [ %4, %ne ]		; <i1> [#uses=1]
	br i1 %5, label %yes, label %no

yes:		; preds = %join
	store i32 0, i32* inttoptr (i32 20000 to i32*)
	ret i32 5

no:		; preds = %join
	ret i32 20
}
