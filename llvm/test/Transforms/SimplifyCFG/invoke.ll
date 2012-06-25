; RUN: opt < %s -simplifycfg -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare i32 @__gxx_personality_v0(...)
declare void @__cxa_call_unexpected(i8*)
declare i64 @llvm.objectsize.i64(i8*, i1) nounwind readonly


; CHECK: @f1
define i8* @f1() nounwind uwtable ssp {
entry:
; CHECK: call void @llvm.trap()
; CHECK: unreachable
  %call = invoke noalias i8* undef()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i8* %call

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          filter [0 x i8*] zeroinitializer
  %1 = extractvalue { i8*, i32 } %0, 0
  tail call void @__cxa_call_unexpected(i8* %1) noreturn nounwind
  unreachable
}

; CHECK: @f2
define i8* @f2() nounwind uwtable ssp {
entry:
; CHECK: call void @llvm.trap()
; CHECK: unreachable
  %call = invoke noalias i8* null()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i8* %call

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          filter [0 x i8*] zeroinitializer
  %1 = extractvalue { i8*, i32 } %0, 0
  tail call void @__cxa_call_unexpected(i8* %1) noreturn nounwind
  unreachable
}
