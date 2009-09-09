; RUN: llc < %s -march=bfin -verify-machineinstrs
@.str_1 = external constant [42 x i8]		; <[42 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define i32 @main(i32 %argc.1, i8** %argv.1) {
entry:
	%tmp.16 = call i32 (i8*, ...)* @printf(i8* getelementptr ([42 x i8]* @.str_1, i64 0, i64 0), i32 0, i32 0, i64 0, i64 0)
	ret i32 0
}
