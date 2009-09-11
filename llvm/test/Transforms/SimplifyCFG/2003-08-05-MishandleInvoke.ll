; Do not remove the invoke!
;
; RUN: opt < %s -simplifycfg -S | grep invoke

define i32 @test() {
	invoke i32 @test( )
			to label %Ret unwind label %Ret		; <i32>:1 [#uses=0]
Ret:		; preds = %0, %0
	%A = add i32 0, 1		; <i32> [#uses=1]
	ret i32 %A
}

