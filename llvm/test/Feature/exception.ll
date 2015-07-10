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
  cleanupret i8 0 unwind label %bb
bb:
  ret void
}

define void @cleanupret1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  cleanupret void unwind label %bb
bb:
  ret void
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
  catchret label %bb
}

define i8 @catchblock() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %bb2
bb:
  ret i8 %cbv
bb2:
  %cbv = catchblock i8 [i7 4] to label %bb unwind label %bb2
}

define void @terminateblock0() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %bb
bb:
  terminateblock [i7 4] unwind label %bb
}

define void @terminateblock1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  terminateblock [i7 4] unwind to caller
}

define void @cleanupblock() personality i32 (...)* @__gxx_personality_v0 {
entry:
  cleanupblock void [i7 4]
  ret void
}

define void @catchendblock0() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %bb
bb:
  catchendblock unwind label %bb
}

define void @catchendblock1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  catchendblock unwind to caller
}
