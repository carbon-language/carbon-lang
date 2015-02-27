; RUN: opt %s -inline -S | FileCheck %s

declare void @external_func()
declare void @abort()

@exception_inner = external global i8
@exception_outer = external global i8
@condition = external global i1


; Check for a bug in which multiple "resume" instructions in the
; inlined function caused "catch i8* @exception_outer" to appear
; multiple times in the resulting landingpad.

define internal void @inner_multiple_resume() {
  invoke void @external_func()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32 personality i8* null
      catch i8* @exception_inner
  %cond = load i1, i1* @condition
  br i1 %cond, label %resume1, label %resume2
resume1:
  resume i32 1
resume2:
  resume i32 2
}

define void @outer_multiple_resume() {
  invoke void @inner_multiple_resume()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32 personality i8* null
      catch i8* @exception_outer
  resume i32 %lp
}
; CHECK: define void @outer_multiple_resume()
; CHECK: %lp.i = landingpad
; CHECK-NEXT: catch i8* @exception_inner
; CHECK-NEXT: catch i8* @exception_outer
; Check that there isn't another "catch" clause:
; CHECK-NEXT: load


; Check for a bug in which having a "resume" and a "call" in the
; inlined function caused "catch i8* @exception_outer" to appear
; multiple times in the resulting landingpad.

define internal void @inner_resume_and_call() {
  call void @external_func()
  invoke void @external_func()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32 personality i8* null
      catch i8* @exception_inner
  resume i32 %lp
}

define void @outer_resume_and_call() {
  invoke void @inner_resume_and_call()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32 personality i8* null
      catch i8* @exception_outer
  resume i32 %lp
}
; CHECK: define void @outer_resume_and_call()
; CHECK: %lp.i = landingpad
; CHECK-NEXT: catch i8* @exception_inner
; CHECK-NEXT: catch i8* @exception_outer
; Check that there isn't another "catch" clause:
; CHECK-NEXT: br


; Check what happens if the inlined function contains an "invoke" but
; no "resume".  In this case, the inlined landingpad does not need to
; include the "catch i8* @exception_outer" clause from the outer
; function (since the outer function's landingpad will not be
; reachable), but it's OK to include this clause.

define internal void @inner_no_resume_or_call() {
  invoke void @external_func()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32 personality i8* null
      catch i8* @exception_inner
  ; A landingpad might have no "resume" if a C++ destructor aborts.
  call void @abort() noreturn nounwind
  unreachable
}

define void @outer_no_resume_or_call() {
  invoke void @inner_no_resume_or_call()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32 personality i8* null
      catch i8* @exception_outer
  resume i32 %lp
}
; CHECK: define void @outer_no_resume_or_call()
; CHECK: %lp.i = landingpad
; CHECK-NEXT: catch i8* @exception_inner
; CHECK-NEXT: catch i8* @exception_outer
; Check that there isn't another "catch" clause:
; CHECK-NEXT: call void @abort()
