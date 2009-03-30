; RUN: llvm-as < %s | opt -loop-index-split | llvm-dis | not grep undef
define i32 @main() {
entry:
	br label %header

header:
	%r = phi i32 [ 0, %entry ], [ %r3, %skip ]
	%i = phi i32 [ 0, %entry ], [ %i1, %skip ]
        %i99 = add i32 %i, 99
	%cond = icmp eq i32 %i99, 3
        br i1 %cond, label %body, label %skip

body:
        br label %skip

skip:
        %r3 = phi i32 [ %r, %header ], [ 3, %body ]
        %i1 = add i32 %i, 1
        %exitcond = icmp eq i32 %i1, 10
        br i1 %exitcond, label %exit, label %header

exit:
        ret i32 %r3
}
