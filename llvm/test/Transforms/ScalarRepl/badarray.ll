; RUN: opt < %s -scalarrepl -instcombine -S | not grep alloca
; PR3466

define i32 @test() {
	%X = alloca [4 x i32]		; <[4 x i32]*> [#uses=1]
        ; Off end of array!
	%Y = getelementptr [4 x i32]* %X, i64 0, i64 6		; <i32*> [#uses=2]
	store i32 0, i32* %Y
	%Z = load i32* %Y		; <i32> [#uses=1]
	ret i32 %Z
}


define i32 @test2() nounwind {
entry:
        %yx2.i = alloca float, align 4          ; <float*> [#uses=1]            
        %yx26.i = bitcast float* %yx2.i to i64*         ; <i64*> [#uses=1]      
        %0 = load i64* %yx26.i, align 8         ; <i64> [#uses=0]               
        unreachable
}
