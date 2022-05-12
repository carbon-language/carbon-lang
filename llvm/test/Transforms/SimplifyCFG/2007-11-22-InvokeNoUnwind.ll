; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; CHECK-NOT: invoke

declare i32 @func(i8*) nounwind

define i32 @test() personality i32 (...)* @__gxx_personality_v0 {
	invoke i32 @func( i8* null )
			to label %Cont unwind label %Other		; <i32>:1 [#uses=0]

Cont:		; preds = %0
	ret i32 0

Other:		; preds = %0
	landingpad { i8*, i32 }
		catch i8* null
	ret i32 1
}

declare i32 @__gxx_personality_v0(...)
