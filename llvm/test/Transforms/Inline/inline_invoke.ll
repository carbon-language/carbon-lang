; RUN: opt < %s -inline -S | FileCheck %s

; Test that the inliner correctly handles inlining into invoke sites
; by appending selectors and forwarding _Unwind_Resume directly to the
; enclosing landing pad.

;; Test 0 - basic functionality.

%struct.A = type { i8 }

@_ZTIi = external constant i8*

declare void @_ZN1AC1Ev(%struct.A*)

declare void @_ZN1AD1Ev(%struct.A*)

declare void @use(i32) nounwind

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


;; Test 1 - Correctly handle phis in outer landing pads.

define void @test1_out() uwtable ssp {
entry:
  invoke void @test0_in()
          to label %cont unwind label %lpad

cont:
  invoke void @test0_in()
          to label %ret unwind label %lpad

ret:
  ret void

lpad:
  %x = phi i32 [ 0, %entry ], [ 1, %cont ]
  %y = phi i32 [ 1, %entry ], [ 4, %cont ]
  %exn = call i8* @llvm.eh.exception() nounwind
  %eh.selector = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %exn, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* bitcast (i8** @_ZTIi to i8*)) nounwind
  %0 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) nounwind
  %1 = icmp eq i32 %eh.selector, %0
  br i1 %1, label %catch, label %eh.resume

catch:
  %ignored = call i8* @__cxa_begin_catch(i8* %exn) nounwind
  call void @use(i32 %x)
  call void @use(i32 %y)
  call void @__cxa_end_catch() nounwind
  br label %ret

eh.resume:
  call void @llvm.eh.resume(i8* %exn, i32 %eh.selector) noreturn
  unreachable
}

; CHECK:    define void @test1_out()
; CHECK:      [[A2:%.*]] = alloca %struct.A,
; CHECK:      [[B2:%.*]] = alloca %struct.A,
; CHECK:      [[A1:%.*]] = alloca %struct.A,
; CHECK:      [[B1:%.*]] = alloca %struct.A,
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[A1]])
; CHECK-NEXT:   unwind label %[[LPAD:[^\s]+]]
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[B1]])
; CHECK-NEXT:   unwind label %[[LPAD1:[^\s]+]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[B1]])
; CHECK-NEXT:   unwind label %[[LPAD1]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[A1]])
; CHECK-NEXT:   unwind label %[[LPAD]]

; Inner landing pad from first inlining.
; CHECK:    [[LPAD1]]:
; CHECK-NEXT: [[EXN1:%.*]] = call i8* @llvm.eh.exception()
; CHECK-NEXT: [[SEL1:%.*]] = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* [[EXN1]], i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 0, i8* bitcast (i8** @_ZTIi to i8*))
; CHECK-NEXT: invoke void @_ZN1AD1Ev(%struct.A* [[A1]])
; CHECK-NEXT:   to label %[[RESUME1:[^\s]+]] unwind
; CHECK: [[RESUME1]]:
; CHECK-NEXT: br label %[[LPAD_JOIN1:[^\s]+]]

; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[A2]])
; CHECK-NEXT:   unwind label %[[LPAD]]
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[B2]])
; CHECK-NEXT:   unwind label %[[LPAD2:[^\s]+]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[B2]])
; CHECK-NEXT:   unwind label %[[LPAD2]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[A2]])
; CHECK-NEXT:   unwind label %[[LPAD]]

; Inner landing pad from second inlining.
; CHECK:    [[LPAD2]]:
; CHECK-NEXT: [[EXN2:%.*]] = call i8* @llvm.eh.exception()
; CHECK-NEXT: [[SEL2:%.*]] = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* [[EXN2]], i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 0, i8* bitcast (i8** @_ZTIi to i8*))
; CHECK-NEXT: invoke void @_ZN1AD1Ev(%struct.A* [[A2]])
; CHECK-NEXT:   to label %[[RESUME2:[^\s]+]] unwind
; CHECK: [[RESUME2]]:
; CHECK-NEXT: br label %[[LPAD_JOIN2:[^\s]+]]

; CHECK:      ret void

; CHECK:    [[LPAD]]:
; CHECK-NEXT: [[X:%.*]] = phi i32 [ 0, %entry ], [ 0, {{%.*}} ], [ 1, %cont ], [ 1, {{%.*}} ]
; CHECK-NEXT: [[Y:%.*]] = phi i32 [ 1, %entry ], [ 1, {{%.*}} ], [ 4, %cont ], [ 4, {{%.*}} ]
; CHECK-NEXT: [[EXN:%.*]] = call i8* @llvm.eh.exception()
; CHECK-NEXT: [[SEL:%.*]] = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* [[EXN]], i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* bitcast (i8** @_ZTIi to i8*))
; CHECK-NEXT: br label %[[LPAD_JOIN2]]

; CHECK: [[LPAD_JOIN2]]:
; CHECK-NEXT: [[XJ2:%.*]] = phi i32 [ [[X]], %[[LPAD]] ], [ 1, %[[RESUME2]] ]
; CHECK-NEXT: [[YJ2:%.*]] = phi i32 [ [[Y]], %[[LPAD]] ], [ 4, %[[RESUME2]] ]
; CHECK-NEXT: [[EXNJ2:%.*]] = phi i8* [ [[EXN]], %[[LPAD]] ], [ [[EXN2]], %[[RESUME2]] ]
; CHECK-NEXT: [[SELJ2:%.*]] = phi i32 [ [[SEL]], %[[LPAD]] ], [ [[SEL2]], %[[RESUME2]] ]
; CHECK-NEXT: br label %[[LPAD_JOIN1]]

; CHECK: [[LPAD_JOIN1]]:
; CHECK-NEXT: [[XJ1:%.*]] = phi i32 [ [[XJ2]], %[[LPAD_JOIN2]] ], [ 0, %[[RESUME1]] ]
; CHECK-NEXT: [[YJ1:%.*]] = phi i32 [ [[YJ2]], %[[LPAD_JOIN2]] ], [ 1, %[[RESUME1]] ]
; CHECK-NEXT: [[EXNJ1:%.*]] = phi i8* [ [[EXNJ2]], %[[LPAD_JOIN2]] ], [ [[EXN1]], %[[RESUME1]] ]
; CHECK-NEXT: [[SELJ1:%.*]] = phi i32 [ [[SELJ2]], %[[LPAD_JOIN2]] ], [ [[SEL1]], %[[RESUME1]] ]
; CHECK-NEXT: [[T:%.*]] = call i32 @llvm.eh.typeid.for(
; CHECK-NEXT: icmp eq i32 [[SELJ1]], [[T]]

; CHECK:      call void @use(i32 [[XJ1]])
; CHECK:      call void @use(i32 [[YJ1]])

; CHECK:      call void @llvm.eh.resume(i8* [[EXNJ1]], i32 [[SELJ1]])

;; Test 2 - Don't make invalid IR for inlines into landing pads without eh.exception calls

define void @test2_out() uwtable ssp {
entry:
  invoke void @test0_in()
          to label %ret unwind label %lpad

ret:
  ret void

lpad:
  call void @_ZSt9terminatev()
  unreachable
}

; CHECK: define void @test2_out()
; CHECK:      [[A:%.*]] = alloca %struct.A,
; CHECK:      [[B:%.*]] = alloca %struct.A,
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[A]])
; CHECK-NEXT:   unwind label %[[LPAD:[^\s]+]]
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[B]])
; CHECK-NEXT:   unwind label %[[LPAD2:[^\s]+]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[B]])
; CHECK-NEXT:   unwind label %[[LPAD2]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[A]])
; CHECK-NEXT:   unwind label %[[LPAD]]
