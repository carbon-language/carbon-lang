; RUN: llc < %s | FileCheck %s
; rdar://problem/6920088
;target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-apple-darwin9.0"
@"\01LC" = internal constant [2 x i8] c"a\00"		; <[2 x i8]*> [#uses=1]
@"\01LC1" = internal constant [2 x i8] c"b\00"		; <[2 x i8]*> [#uses=1]
@"\01LC2" = internal constant [2 x i8] c"c\00"		; <[2 x i8]*> [#uses=1]
@"\01LC3" = internal constant [2 x i8] c"d\00"		; <[2 x i8]*> [#uses=1]
@"\01LC4" = internal constant [2 x i8] c"e\00"		; <[2 x i8]*> [#uses=1]
@"\01LC5" = internal constant [2 x i8] c"f\00"		; <[2 x i8]*> [#uses=1]
@"\01LC6" = internal constant [2 x i8] c"g\00"		; <[2 x i8]*> [#uses=1]
@"\01LC7" = internal constant [4 x i8] c"%s\0A\00"		; <[4 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	%tmp = alloca i8*		; <i8**> [#uses=2]
	%tmp1 = alloca i8*		; <i8**> [#uses=2]
	%tmp2 = alloca i8*		; <i8**> [#uses=2]
; CHECK:      leaq LC4(%rip), [[AREG:%[a-z]+]]
; CHECK-NEXT: movq [[AREG]], [[STKOFF:[0-9]+]](%rsp)
	store i8* getelementptr ([2 x i8]* @"\01LC4", i32 0, i32 0), i8** %tmp
	store i8* getelementptr ([2 x i8]* @"\01LC5", i32 0, i32 0), i8** %tmp1
	store i8* getelementptr ([2 x i8]* @"\01LC6", i32 0, i32 0), i8** %tmp2
; The LC4 struct should be passed in %r9:
; CHECK:      movq [[STKOFF]](%rsp), %r9
	call void (i8**,  ...)* @generate_password(i8** null, 
         i8* getelementptr ([2 x i8]* @"\01LC", i32 0, i32 0),
         i8* getelementptr ([2 x i8]* @"\01LC1", i32 0, i32 0),
         i8* getelementptr ([2 x i8]* @"\01LC2", i32 0, i32 0),
         i8* getelementptr ([2 x i8]* @"\01LC3", i32 0, i32 0),
         i8** byval %tmp, i8** byval %tmp1, i8** byval %tmp2)
	ret i32 0
}

declare void @generate_password(i8** %pw, ...) nounwind
