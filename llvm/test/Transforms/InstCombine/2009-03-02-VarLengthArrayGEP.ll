; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {getelementptr i32}
; PR3694

define i32 @e(i32 %m, i32 %n) nounwind {
entry:
	%0 = alloca i32, i32 %n, align 4		; <i32*> [#uses=2]
	%1 = bitcast i32* %0 to [0 x i32]*		; <[0 x i32]*> [#uses=1]
	call void @f(i32* %0) nounwind
	%2 = getelementptr [0 x i32]* %1, i32 0, i32 %m		; <i32*> [#uses=1]
	%3 = load i32* %2, align 4		; <i32> [#uses=1]
	ret i32 %3
}

declare void @f(i32*)
