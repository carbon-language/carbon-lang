; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following code:
;
; class Obj {
; public:
;   ~Obj();
; };
;
; void test(void)
; {
;   try {
;     Obj o;
;     throw 1;
;   } catch (...) {
;     throw;
;   }
; }

; ModuleID = 'cppeh-catch-and-throw.cpp'
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }
%class.Obj = type { i8 }

$"\01??_R0H@8" = comdat any

$"_CT??_R0H@84" = comdat any

$_CTA1H = comdat any

$_TI1H = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@__ImageBase = external constant i8
@"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
@_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableType* @"_CT??_R0H@84" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableTypeArray.1* @_CTA1H to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section ".xdata", comdat

; This is just a minimal check to verify that main was handled by WinEHPrepare.
; CHECK: define void @"\01?test@@YAXXZ"()
; CHECK: entry:
; CHECK:   call void (...) @llvm.localescape
; CHECK:   invoke void @_CxxThrowException
; CHECK: }

; Function Attrs: uwtable
define void @"\01?test@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %o = alloca %class.Obj, align 1
  %tmp = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 1, i32* %tmp
  %0 = bitcast i32* %tmp to i8*
  invoke void @_CxxThrowException(i8* %0, %eh.ThrowInfo* @_TI1H) #3
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  store i8* %2, i8** %exn.slot
  %3 = extractvalue { i8*, i32 } %1, 1
  store i32 %3, i32* %ehselector.slot
  call void @"\01??1Obj@@QEAA@XZ"(%class.Obj* %o) #2
  br label %catch

catch:                                            ; preds = %lpad
  %exn = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn, i8* null) #2
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) #3
          to label %unreachable unwind label %lpad1

lpad1:                                            ; preds = %catch
  %4 = landingpad { i8*, i32 }
          cleanup
  %5 = extractvalue { i8*, i32 } %4, 0
  store i8* %5, i8** %exn.slot
  %6 = extractvalue { i8*, i32 } %4, 1
  store i32 %6, i32* %ehselector.slot
  call void @llvm.eh.endcatch() #2
  br label %eh.resume

try.cont:                                         ; No predecessors!
  ret void

eh.resume:                                        ; preds = %lpad1
  %exn2 = load i8*, i8** %exn.slot
  %sel = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn2, 0
  %lpad.val3 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1
  resume { i8*, i32 } %lpad.val3

unreachable:                                      ; preds = %catch, %entry
  unreachable
}

; Verify that we inserted a stub invoke into the outlined cleanup handler.
;
; CHECK-LABEL: define internal void @"\01?test@@YAXXZ.cleanup"(i8*, i8*)
; CHECK: entry:
; CHECK:   call i8* @llvm.localrecover
; CHECK:   call void @"\01??1Obj@@QEAA@XZ"
; CHECK:   invoke void @llvm.donothing()
; CHECK:           to label %[[SPLIT_LABEL:.+]] unwind label %[[LPAD_LABEL:.+]]
;
; CHECK: [[SPLIT_LABEL]]
;
; CHECK: [[LPAD_LABEL]]
; CHECK:   landingpad { i8*, i32 }
; CHECK:           cleanup
; CHECK:   unreachable
; CHECK: }

declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind
declare void @"\01??1Obj@@QEAA@XZ"(%class.Obj*) #1

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #2

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (trunk 235214) (llvm/trunk 235213)"}
