; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s
; This testcase makes sure we can handle invoke @llvm.donothing without
; assertion failure.
; rdar://problem/13228754
; CHECK: .globl  _main

declare void @callA()
declare i32 @__gxx_personality_sj0(...)

define void @main() {
invoke.cont:
  invoke void @callA() 
          to label %invoke.cont25 unwind label %lpad2
invoke.cont25:
  invoke void @llvm.donothing()
          to label %invoke.cont27 unwind label %lpad15

invoke.cont27:
  invoke void @callB()
          to label %invoke.cont75 unwind label %lpad15

invoke.cont75:
  ret void

lpad2:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*)
          cleanup
  br label %eh.resume

lpad15:
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*)
          cleanup
  br label %eh.resume

eh.resume:
  resume { i8*, i32 } zeroinitializer
}

declare void @callB()
declare void @llvm.donothing() nounwind readnone
