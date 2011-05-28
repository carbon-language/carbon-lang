; RUN: opt < %s -inline -S | FileCheck %s

; Test that the inliner correctly handles inlining into invoke sites
; by appending selectors and forwarding _Unwind_Resume directly to the
; enclosing landing pad.

%struct.A = type { i8 }

@_ZTIi = external constant i8*

declare void @_ZN1AC1Ev(%struct.A*)

declare void @_ZN1AD1Ev(%struct.A*)

declare i8* @llvm.eh.exception() nounwind readonly

declare i32 @llvm.eh.selector(i8*, i8*, ...) nounwind

declare i32 @llvm.eh.typeid.for(i8*) nounwind

declare void @llvm.eh.resume(i8*, i32)

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

declare void @_ZSt9terminatev()

define internal void @test0_in() alwaysinline uwtable ssp {
entry:
  %a = alloca %struct.A, align 1
  %b = alloca %struct.A, align 1
  call void @_ZN1AC1Ev(%struct.A* %a)
  invoke void @_ZN1AC1Ev(%struct.A* %b)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  invoke void @_ZN1AD1Ev(%struct.A* %b)
          to label %invoke.cont1 unwind label %lpad

invoke.cont1:
  call void @_ZN1AD1Ev(%struct.A* %a)
  ret void

lpad:
  %exn = call i8* @llvm.eh.exception() nounwind
  %eh.selector = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %exn, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 0) nounwind
  invoke void @_ZN1AD1Ev(%struct.A* %a)
          to label %invoke.cont2 unwind label %terminate.lpad

invoke.cont2:
  call void @llvm.eh.resume(i8* %exn, i32 %eh.selector) noreturn
  unreachable

terminate.lpad:
  %exn3 = call i8* @llvm.eh.exception() nounwind
  %eh.selector4 = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %exn3, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null) nounwind
  call void @_ZSt9terminatev() noreturn nounwind
  unreachable
}

define void @test0_out() uwtable ssp {
entry:
  invoke void @test0_in()
          to label %ret unwind label %lpad

ret:
  ret void

lpad:                                             ; preds = %entry
  %exn = call i8* @llvm.eh.exception() nounwind
  %eh.selector = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %exn, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* bitcast (i8** @_ZTIi to i8*)) nounwind
  %0 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) nounwind
  %1 = icmp eq i32 %eh.selector, %0
  br i1 %1, label %catch, label %eh.resume

catch:
  %ignored = call i8* @__cxa_begin_catch(i8* %exn) nounwind
  call void @__cxa_end_catch() nounwind
  br label %ret

eh.resume:
  call void @llvm.eh.resume(i8* %exn, i32 %eh.selector) noreturn
  unreachable
}

; CHECK:    define void @test0_out()
; CHECK:      [[A:%.*]] = alloca %struct.A,
; CHECK:      [[B:%.*]] = alloca %struct.A,
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[A]])
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[B]])
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[B]])
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[A]])
; CHECK:      call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* {{%.*}}, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 0, i8* bitcast (i8** @_ZTIi to i8*))
; CHECK-NEXT: invoke void @_ZN1AD1Ev(%struct.A* [[A]])
; CHECK-NEXT:   to label %[[LBL:[^\s]+]] unwind
; CHECK: [[LBL]]:
; CHECK-NEXT: br label %[[LPAD:[^\s]+]]
; CHECK:      ret void
; CHECK:      call i8* @llvm.eh.exception()
; CHECK-NEXT: call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* {{%.*}}, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* bitcast (i8** @_ZTIi to i8*))
; CHECK-NEXT: br label %[[LPAD]]
; CHECK: [[LPAD]]:
; CHECK-NEXT: phi i8* [
; CHECK-NEXT: phi i32 [
; CHECK-NEXT: call i32 @llvm.eh.typeid.for(