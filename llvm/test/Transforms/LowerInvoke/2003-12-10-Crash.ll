; This testcase was reduced from Shootout-C++/reversefile.cpp by bugpoint

; RUN: opt < %s -lowerinvoke -disable-output

declare void @baz()

declare void @bar()

define void @foo() personality i32 (...)* @__gxx_personality_v0 {
then:
	invoke void @baz( )
			to label %invoke_cont.0 unwind label %try_catch
invoke_cont.0:		; preds = %then
	invoke void @bar( )
			to label %try_exit unwind label %try_catch
try_catch:		; preds = %invoke_cont.0, %then
	%__tmp.0 = phi i32* [ null, %invoke_cont.0 ], [ null, %then ]		; <i32*> [#uses=0]
  %res = landingpad { i8* }
          cleanup
	ret void
try_exit:		; preds = %invoke_cont.0
	ret void
}

declare i32 @__gxx_personality_v0(...)
