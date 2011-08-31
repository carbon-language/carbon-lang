; RUN: opt < %s -prune-eh -S | grep invoke

declare void @External()

define void @foo() {
	invoke void @External( )
			to label %Cont unwind label %Cont
Cont:		; preds = %0, %0
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
	ret void
}

declare i32 @__gxx_personality_v0(...)
