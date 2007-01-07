; RUN: llvm-as < %s | opt -indvars -disable-output &&
; RUN: llvm-as < %s | opt -indvars | llvm-dis | not grep 'ret i32 0'
; PR1015

target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-apple-darwin8"
%foo = internal constant [5 x i8] c"\00abc\00"		; <[5 x i8]*> [#uses=1]
%str = internal constant [4 x i8] c"%d\0A\00"		; <[4 x i8]*> [#uses=1]

implementation   ; Functions:

define i32 %test(i32 %J) {
entry:
	br label %bb2

bb:		; preds = %cond_next, %cond_true
	%tmp1 = add i32 %i.0, 1		; <i32> [#uses=1]
	br label %bb2

bb2:		; preds = %bb, %entry
	%i.0 = phi i32 [ 0, %entry ], [ %tmp1, %bb ]		; <i32> [#uses=4]
	%tmp = icmp eq i32 %i.0, 0		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %cond_next

cond_true:		; preds = %bb2
	br label %bb

cond_next:		; preds = %bb2
	%tmp = getelementptr [5 x i8]* %foo, i32 0, i32 %i.0		; <i8*> [#uses=1]
	%tmp = load i8* %tmp		; <i8> [#uses=1]
	%tmp5 = icmp eq i8 %tmp, 0		; <bool> [#uses=1]
	br bool %tmp5, label %bb6, label %bb

bb6:		; preds = %cond_next
	br label %return

return:		; preds = %bb6
	ret i32 %i.0
}

define void %main() {
entry:
	%tmp = call i32 %test( i32 0 )		; <i32> [#uses=1]
	%tmp1 = call i32 (i8*, ...)* %printf( i8* getelementptr ([4 x i8]* %str, i32 0, i32 0), i32 %tmp )		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

declare i32 %printf(i8*, ...)
