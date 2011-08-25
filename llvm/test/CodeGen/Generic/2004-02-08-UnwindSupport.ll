; RUN: llc < %s -enable-correct-eh-support

define i32 @test() {
        unwind
}

define i32 @main() {
        %X = invoke i32 @test( )
                        to label %cont unwind label %EH         ; <i32> [#uses=0]

cont:           ; preds = %0
        ret i32 1

EH:             ; preds = %0
  %lpad = landingpad { i8*, i32 } personality i32 (...)* @__gxx_personality_v0
            cleanup
  ret i32 0
}

declare i32 @__gxx_personality_v0(...)
