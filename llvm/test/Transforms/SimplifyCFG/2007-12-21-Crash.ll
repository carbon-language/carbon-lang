; RUN:  llvm-as < %s | opt -std-compile-opts -disable-output

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

define i32 @bork() {
entry:
	%retval = alloca i32		; <i32*> [#uses=2]
	%opt = alloca i32		; <i32*> [#uses=3]
	%undo = alloca i32		; <i32*> [#uses=3]
	%tmp = alloca i32		; <i32*> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 0, i32* %undo, align 4
	br label %bb5

bb:		; preds = %bb5
	%tmp1 = load i32* %opt, align 4		; <i32> [#uses=1]
	switch i32 %tmp1, label %bb4 [
		 i32 102, label %bb3
		 i32 110, label %bb2
	]

bb2:		; preds = %bb
	store i32 1, i32* %undo, align 4
	br label %bb3

bb3:		; preds = %bb2, %bb
	br label %bb5

bb4:		; preds = %bb
	store i32 258, i32* %tmp, align 4
	br label %bb13

bb5:		; preds = %bb3, %entry
	%tmp6 = call i32 (...)* @foo( ) nounwind 		; <i32> [#uses=1]
	store i32 %tmp6, i32* %opt, align 4
	%tmp7 = load i32* %opt, align 4		; <i32> [#uses=1]
	%tmp8 = icmp ne i32 %tmp7, -1		; <i1> [#uses=1]
	%tmp89 = zext i1 %tmp8 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp89, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb, label %bb10

bb10:		; preds = %bb5
	%tmp11 = load i32* %undo, align 4		; <i32> [#uses=1]
	%tmp12 = call i32 (...)* @bar( i32 %tmp11 ) nounwind 		; <i32> [#uses=0]
	store i32 1, i32* %tmp, align 4
	br label %bb13

bb13:		; preds = %bb10, %bb4
	%tmp14 = load i32* %tmp, align 4		; <i32> [#uses=1]
	store i32 %tmp14, i32* %retval, align 4
	br label %return

return:		; preds = %bb13
	%retval15 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval15
}

declare i32 @foo(...)

declare i32 @bar(...)
