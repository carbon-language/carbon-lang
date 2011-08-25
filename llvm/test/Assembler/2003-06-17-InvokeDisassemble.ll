; RUN: llvm-as < %s | llvm-dis

define void @test() {
  invoke void @test( )
    to label %Next unwind label %Next

Next:           ; preds = %0, %0
  %lpad = landingpad { i8*, i32 } personality i32 (...)* @__gxx_personality_v0
            cleanup
  ret void
}

declare i32 @__gxx_personality_v0(...)
