; RUN: opt -S -jump-threading < %s -disable-output
; PR17621
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

declare void @_ZN1F7doApplyEv() unnamed_addr readnone align 2

define void @_Z3fn1v() uwtable {
entry:
  store i32 0, i32* undef, align 4
  invoke void @_ZN1F7doApplyEv()
          to label %_ZN1D5applyEv.exit unwind label %lpad1

_ZN1D5applyEv.exit:
  invoke void @_ZN1F10insertTextEv()
          to label %invoke.cont7 unwind label %lpad1

invoke.cont7:
  ret void

lpad1:
  %tmp1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  %tmp2 = load i32* undef, align 4
  %tobool.i.i.i = icmp eq i32 %tmp2, 0
  br i1 %tobool.i.i.i, label %_ZN1BI1FED1Ev.exit, label %if.then.i.i.i

if.then.i.i.i:
  br i1 undef, label %_ZN1BI1FED1Ev.exit, label %delete.notnull.i.i.i

delete.notnull.i.i.i:
  unreachable

_ZN1BI1FED1Ev.exit:
  br label %eh.resume

eh.resume:
  resume { i8*, i32 } undef
}

declare i32 @__gxx_personality_v0(...)

declare void @_ZN1F10insertTextEv()
