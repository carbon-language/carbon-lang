; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i32 %A}

define i32 @test(i32 %A) {
	%X = or i1 false, false		
	br i1 %X, label %T, label %C

T:		; preds = %0
	%B = add i32 %A, 1	
	br label %C

C:		; preds = %T, %0
	%C.upgrd.1 = phi i32 [ %B, %T ], [ %A, %0 ]
	ret i32 %C.upgrd.1
}
