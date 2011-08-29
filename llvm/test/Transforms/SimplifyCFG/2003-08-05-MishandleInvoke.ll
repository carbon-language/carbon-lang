; Do not remove the invoke!
;
; RUN: opt < %s -simplifycfg -S | grep invoke

define i32 @test() {
	invoke i32 @test( )
			to label %Ret unwind label %Ret		; <i32>:1 [#uses=0]
Ret:		; preds = %0, %0
        %val = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
                 catch i8* null
	%A = add i32 0, 1		; <i32> [#uses=1]
	ret i32 %A
}

declare i32 @__gxx_personality_v0(...)
