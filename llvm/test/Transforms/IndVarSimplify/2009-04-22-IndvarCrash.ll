; RUN: llvm-as < %s | opt -indvars

define i32 @t() nounwind ssp {
entry:
	br label %bb32

bb32:		; preds = %bb32, %entry
	%mbPartIdx.0.reg2mem.0 = phi i8 [ %2, %bb32 ], [ 0, %entry ]		; <i8> [#uses=3]
	%0 = and i8 %mbPartIdx.0.reg2mem.0, 1		; <i8> [#uses=0]
	%1 = zext i8 %mbPartIdx.0.reg2mem.0 to i64		; <i64> [#uses=0]
	%2 = add i8 %mbPartIdx.0.reg2mem.0, 1		; <i8> [#uses=2]
	%3 = icmp ugt i8 %2, 3		; <i1> [#uses=1]
	br i1 %3, label %bb41, label %bb32

bb41:		; preds = %bb32
	ret i32 0
}
