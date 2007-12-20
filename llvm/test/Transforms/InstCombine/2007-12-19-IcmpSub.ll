; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {sub}

define i32 @foo(i32 %a) {
entry:
	%tmp2 = sub i32 99, %a		; <i32> [#uses=1]
	%tmp3 = icmp sgt i32 %tmp2, -1		; <i1> [#uses=1]
	%retval = select i1 %tmp3, i32 %a, i32 0		; <i32> [#uses=1]
	ret i32 %retval
}

define i32 @bar(i32 %a) {
entry:
	%tmp2 = sub i32 99, %a		; <i32> [#uses=1]
	%tmp3 = icmp sge i32 %tmp2, 0; <i1> [#uses=1]
	%retval = select i1 %tmp3, i32 %a, i32 0		; <i32> [#uses=1]
	ret i32 %retval
}

define i32 @baz(i32 %a) {
entry:
	%tmp2 = sub i32 99, %a		; <i32> [#uses=1]
	%tmp3 = icmp slt i32 %tmp2, 1		; <i1> [#uses=1]
	%retval = select i1 %tmp3, i32 %a, i32 0		; <i32> [#uses=1]
	ret i32 %retval
}