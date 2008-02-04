; RUN: llvm-as < %s | llc

@.str3 = external constant [56 x i8]		; <[56 x i8]*> [#uses=1]

define i32 @main() nounwind  {
entry:
	br label %bb30

bb30:		; preds = %bb30, %entry
	%l.024 = phi i64 [ -10000, %entry ], [ 0, %bb30 ]		; <i64> [#uses=2]
	%tmp37 = tail call i64 @llvm.ctlz.i64( i64 %l.024 )		; <i64> [#uses=1]
	trunc i64 %tmp37 to i32		; <i32>:0 [#uses=1]
	%tmp40 = tail call i32 (i8*, ...)* @printf( i8* noalias  getelementptr ([56 x i8]* @.str3, i32 0, i32 0), i64 %l.024, i32 %0, i32 0, i32 0 ) nounwind 		; <i32> [#uses=0]
	br i1 false, label %bb30, label %bb9.i

bb9.i:		; preds = %bb30
	ret i32 0
}

declare i32 @printf(i8* noalias , ...) nounwind 

declare i64 @llvm.ctlz.i64(i64) nounwind readnone 
