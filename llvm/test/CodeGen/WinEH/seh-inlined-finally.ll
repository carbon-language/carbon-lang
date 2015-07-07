; RUN: opt -S -winehprepare < %s | FileCheck %s

; Check that things work when the mid-level optimizer inlines the finally
; block.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%struct._RTL_CRITICAL_SECTION = type { %struct._RTL_CRITICAL_SECTION_DEBUG*, i32, i32, i8*, i8*, i64 }
%struct._RTL_CRITICAL_SECTION_DEBUG = type { i16, i16, %struct._RTL_CRITICAL_SECTION*, %struct._LIST_ENTRY, i32, i32, i32, i16, i16 }
%struct._LIST_ENTRY = type { %struct._LIST_ENTRY*, %struct._LIST_ENTRY* }

declare i32 @puts(i8*)
declare void @may_crash()
declare i32 @__C_specific_handler(...)
declare i8* @llvm.localrecover(i8*, i8*, i32) #1
declare i8* @llvm.localaddress()
declare void @llvm.localescape(...)
declare dllimport void @EnterCriticalSection(%struct._RTL_CRITICAL_SECTION*)
declare dllimport void @LeaveCriticalSection(%struct._RTL_CRITICAL_SECTION*)

define void @use_finally() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  invoke void @may_crash()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %call.i = tail call i32 @puts(i8* null)
  ret void

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          cleanup
  %call.i2 = tail call i32 @puts(i8* null)
  resume { i8*, i32 } %0
}

; CHECK-LABEL: define void @use_finally()
; CHECK: invoke void @may_crash()
;
; CHECK: landingpad
; CHECK-NEXT: cleanup
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 0, void (i8*, i8*)* @use_finally.cleanup)
; CHECK-NEXT: indirectbr i8* %recover, []

; Function Attrs: nounwind uwtable
define i32 @call_may_crash_locked() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  %p = alloca %struct._RTL_CRITICAL_SECTION, align 8
  call void (...) @llvm.localescape(%struct._RTL_CRITICAL_SECTION* %p)
  call void @EnterCriticalSection(%struct._RTL_CRITICAL_SECTION* %p)
  invoke void @may_crash()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %tmp2 = call i8* @llvm.localaddress()
  %tmp3 = call i8* @llvm.localrecover(i8* bitcast (i32 ()* @call_may_crash_locked to i8*), i8* %tmp2, i32 0) #2
  %tmp6 = bitcast i8* %tmp3 to %struct._RTL_CRITICAL_SECTION*
  call void @LeaveCriticalSection(%struct._RTL_CRITICAL_SECTION* %tmp6)
  ret i32 42

lpad:                                             ; preds = %entry
  %tmp7 = landingpad { i8*, i32 }
            cleanup
  %tmp8 = call i8* @llvm.localaddress()
  %tmp9 = call i8* @llvm.localrecover(i8* bitcast (i32 ()* @call_may_crash_locked to i8*), i8* %tmp8, i32 0)
  %tmp12 = bitcast i8* %tmp9 to %struct._RTL_CRITICAL_SECTION*
  call void @LeaveCriticalSection(%struct._RTL_CRITICAL_SECTION* %tmp12)
  resume { i8*, i32 } %tmp7
}

; CHECK-LABEL: define i32 @call_may_crash_locked()
; CHECK: invoke void @may_crash()
;
; CHECK: landingpad
; CHECK-NEXT: cleanup
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 0, void (i8*, i8*)* @call_may_crash_locked.cleanup)
; CHECK-NEXT: indirectbr i8* %recover, []

; CHECK-LABEL: define internal void @call_may_crash_locked.cleanup(i8*, i8*)
; CHECK: %tmp9 = call i8* @llvm.localrecover(i8* bitcast (i32 ()* @call_may_crash_locked to i8*), i8* %1, i32 0)
; CHECK: %tmp12 = bitcast i8* %tmp9 to %struct._RTL_CRITICAL_SECTION*
; CHECK: call void @LeaveCriticalSection(%struct._RTL_CRITICAL_SECTION* %tmp12)
