; RUN: llc < %s -march=bfin -verify-machineinstrs > %t

@.str = external constant [14 x i8]		; <[14 x i8]*> [#uses=1]

define i32 @main(i64 %arg) nounwind {
entry:
	%tmp47 = tail call i64 @llvm.cttz.i64(i64 %arg)		; <i64> [#uses=1]
	%tmp48 = trunc i64 %tmp47 to i32		; <i32> [#uses=1]
	%tmp40 = tail call i32 (i8*, ...)* @printf(i8* noalias getelementptr ([14 x i8]* @.str, i32 0, i32 0), i64 %arg, i32 0, i32 %tmp48, i32 0) nounwind		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @printf(i8* noalias, ...) nounwind

declare i64 @llvm.cttz.i64(i64) nounwind readnone
