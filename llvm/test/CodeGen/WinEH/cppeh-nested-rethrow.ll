; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test was generated from the following code.
;
; void test1() {
;   try {
;     try {
;       throw 1;
;     } catch(...) { throw; }
;   } catch (...) { }
; }
; void test2() {
;   try {
;     throw 1;
;   } catch(...) {
;     try {
;       throw; 
;     } catch (...) {}
;   }
; }
;
; These two functions result in functionally equivalent code, but the last
; catch block contains a call to llvm.eh.endcatch that tripped up processing
; during development.
;
; The main purpose of this test is to verify that we can correctly
; handle the case of nested landing pads that return directly to a block in
; the parent function.

; ModuleID = 'cppeh-nested-rethrow.cpp'
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }

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

; CHECK-LABEL: define void @"\01?test1@@YAXXZ"()
; CHECK: entry:
; CHECK:   call void (...) @llvm.localescape

; Function Attrs: nounwind uwtable
define void @"\01?test1@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %tmp = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 1, i32* %tmp
  %0 = bitcast i32* %tmp to i8*
  invoke void @_CxxThrowException(i8* %0, %eh.ThrowInfo* @_TI1H) #2
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  store i8* %2, i8** %exn.slot
  %3 = extractvalue { i8*, i32 } %1, 1
  store i32 %3, i32* %ehselector.slot
  br label %catch

catch:                                            ; preds = %lpad
  %exn = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn, i8* null) #1
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) #2
          to label %unreachable unwind label %lpad1

lpad1:                                            ; preds = %catch
  %4 = landingpad { i8*, i32 }
          catch i8* null
  %5 = extractvalue { i8*, i32 } %4, 0
  store i8* %5, i8** %exn.slot
  %6 = extractvalue { i8*, i32 } %4, 1
  store i32 %6, i32* %ehselector.slot
  br label %catch2

catch2:                                           ; preds = %lpad1
  %exn3 = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn3, i8* null) #1
  call void @llvm.eh.endcatch() #1
  br label %try.cont.4

; This block should not be eliminated.
; CHECK: try.cont.4:
try.cont.4:                                        ; preds = %catch2, %try.cont
  ret void

try.cont:                                         ; No predecessors!
  br label %try.cont.4

unreachable:                                      ; preds = %catch, %entry
  unreachable
; CHECK: }
}

declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #1

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #1

; CHECK-LABEL: define void @"\01?test2@@YAXXZ"()
; CHECK: entry:
; CHECK:   call void (...) @llvm.localescape

; Function Attrs: nounwind uwtable
define void @"\01?test2@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %tmp = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 1, i32* %tmp
  %0 = bitcast i32* %tmp to i8*
  invoke void @_CxxThrowException(i8* %0, %eh.ThrowInfo* @_TI1H) #2
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  store i8* %2, i8** %exn.slot
  %3 = extractvalue { i8*, i32 } %1, 1
  store i32 %3, i32* %ehselector.slot
  br label %catch

catch:                                            ; preds = %lpad
  %exn = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn, i8* null) #1
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) #2
          to label %unreachable unwind label %lpad1

lpad1:                                            ; preds = %catch
  %4 = landingpad { i8*, i32 }
          catch i8* null
  %5 = extractvalue { i8*, i32 } %4, 0
  store i8* %5, i8** %exn.slot
  %6 = extractvalue { i8*, i32 } %4, 1
  store i32 %6, i32* %ehselector.slot
  br label %catch2

catch2:                                           ; preds = %lpad1
  %exn3 = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn3, i8* null) #1
  call void @llvm.eh.endcatch() #1
  br label %try.cont

; This block should not be eliminated.
; CHECK: try.cont:
; The endcatch call should be eliminated.
; CHECK-NOT: call void @llvm.eh.endcatch()
try.cont:                                         ; preds = %catch2
  call void @llvm.eh.endcatch() #1
  br label %try.cont.4

try.cont.4:                                        ; preds = %try.cont
  ret void

unreachable:                                      ; preds = %catch, %entry
  unreachable
; CHECK: }
}

; The outlined test1.catch handler should return to a valid block address.
; CHECK-LABEL: define internal i8* @"\01?test1@@YAXXZ.catch"(i8*, i8*)
; CHECK-NOT:  ret i8* inttoptr (i32 1 to i8*)
; CHECK: }

; The outlined test1.catch1 handler should not contain a return instruction.
; CHECK-LABEL: define internal i8* @"\01?test1@@YAXXZ.catch.1"(i8*, i8*)
; CHECK-NOT: ret
; CHECK: }

; The outlined test2.catch handler should return to a valid block address.
; CHECK-LABEL: define internal i8* @"\01?test2@@YAXXZ.catch"(i8*, i8*)
; CHECK-NOT:  ret i8* inttoptr (i32 1 to i8*)
; CHECK: }

; The outlined test2.catch2 handler should not contain a return instruction.
; CHECK-LABEL: define internal i8* @"\01?test2@@YAXXZ.catch.2"(i8*, i8*)
; CHECK-NOT: ret
; CHECK: }


attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (trunk 236059)"}
