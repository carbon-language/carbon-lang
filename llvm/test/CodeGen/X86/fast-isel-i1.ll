; RUN: llvm-as < %s | llc -march=x86 -fast-isel | grep {andb	\$1, %}

declare i64 @bar(i64)

define i32 @foo(i64 %x) nounwind {
	%y = add i64 %x, -3		; <i64> [#uses=1]
	%t = call i64 @bar(i64 %y)		; <i64> [#uses=1]
	%s = mul i64 %t, 77		; <i64> [#uses=1]
	%z = trunc i64 %s to i1		; <i1> [#uses=1]
	br label %next

next:		; preds = %0
	%u = zext i1 %z to i32		; <i32> [#uses=1]
	%v = add i32 %u, 1999		; <i32> [#uses=1]
	br label %exit

exit:		; preds = %next
	ret i32 %v
}
