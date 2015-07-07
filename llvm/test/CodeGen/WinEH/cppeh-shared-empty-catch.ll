; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following source, built with -O2
;
; void f() {
;   try {
;     g();
;     try {
;       throw;
;     } catch (int) {
;     }
;   } catch (...) {
;   }
; }
;

; ModuleID = '<stdin>'
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchHandlerType = type { i32, i8* }
%eh.ThrowInfo = type { i32, i32, i32, i32 }

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@llvm.eh.handlertype.H.0 = private unnamed_addr constant %eh.CatchHandlerType { i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*) }, section "llvm.metadata"

; CHECK-LABEL: define void @"\01?f@@YAXXZ"()
; CHECK: entry:
; CHECK:   call void (...) @llvm.localescape()
; CHECK:   invoke void @"\01?g@@YAXXZ"()

; Function Attrs: nounwind
define void @"\01?f@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @"\01?g@@YAXXZ"()
          to label %invoke.cont unwind label %lpad

; CHECK-LABEL: invoke.cont:
; CHECK:   invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null)
; CHECK:           to label %unreachable unwind label %[[LPAD1_LABEL:lpad[0-9]+]]

invoke.cont:                                      ; preds = %entry
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) #4
          to label %unreachable unwind label %lpad1

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  br label %catch2

; Note: Even though this landing pad has two catch clauses, it only has one action because both
;       handlers do the same thing.
; CHECK: [[LPAD1_LABEL]]:
; CHECK:   landingpad { i8*, i32 }
; CHECK-NEXT:           catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.0
; CHECK-NEXT:           catch i8* null
; CHECK-NEXT:   [[RECOVER:\%.+]] = call i8* (...) @llvm.eh.actions(i32 1, i8* null, i32 -1, i8* (i8*, i8*)* @"\01?f@@YAXXZ.catch")
; CHECK-NEXT:   indirectbr i8* [[RECOVER]], [label %try.cont4]

lpad1:                                            ; preds = %invoke.cont
  %2 = landingpad { i8*, i32 }
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.0
          catch i8* null
  %3 = extractvalue { i8*, i32 } %2, 0
  br label %catch2

catch2:                                           ; preds = %lpad1, %lpad
  %exn.slot.0 = phi i8* [ %3, %lpad1 ], [ %1, %lpad ]
  tail call void @llvm.eh.begincatch(i8* %exn.slot.0, i8* null) #3
  tail call void @llvm.eh.endcatch() #3
  br label %try.cont4

try.cont4:                                        ; preds = %catch, %catch2
  ret void

unreachable:                                      ; preds = %invoke.cont
  unreachable

; CHECK: }
}

declare void @"\01?g@@YAXXZ"() #1

declare i32 @__CxxFrameHandler3(...)

declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #3

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }
attributes #4 = { noreturn }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 235112) (llvm/trunk 235121)"}
