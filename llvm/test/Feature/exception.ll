; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@_ZTIc = external constant i8*
@_ZTId = external constant i8*
@_ZTIPKc = external constant i8*

define void @_Z3barv() uwtable optsize ssp personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %lpad

try.cont:                                         ; preds = %entry, %invoke.cont4
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {i8*, i32}
            cleanup
            catch i8** @_ZTIc
            filter [2 x i8**] [i8** @_ZTIPKc, i8** @_ZTId]
  resume { i8*, i32 } %exn
}

declare void @_Z3quxv() optsize

declare i32 @__gxx_personality_v0(...)

define void @cleanupret0() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  cleanuppad void [i7 4]
  cleanupret i8 0 unwind label %bb
}

define void @cleanupret1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  cleanuppad void [i7 4]
  cleanupret void unwind label %bb
}

define void @cleanupret2() personality i32 (...)* @__gxx_personality_v0 {
entry:
  cleanupret i8 0 unwind to caller
}

define void @cleanupret3() personality i32 (...)* @__gxx_personality_v0 {
  cleanupret void unwind to caller
}

define void @catchret() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %bb
bb:
  catchret void to label %bb
}

define i8 @catchpad() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %bb2
bb:
  catchret token %cbv to label %exit

exit:
  ret i8 0
bb2:
  %cbv = catchpad token [i7 4] to label %bb unwind label %bb3
bb3:
  catchendpad unwind to caller
}

define void @terminatepad0() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  terminatepad [i7 4] unwind label %bb
}

define void @terminatepad1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  terminatepad [i7 4] unwind to caller
}

define void @cleanuppad() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  cleanuppad void [i7 4]
  ret void
}

define void @catchendpad0() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  catchendpad unwind label %bb
}

define void @catchendpad1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  catchendpad unwind to caller
}
