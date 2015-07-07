; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following code:
;
;void test()
;{
;  try {
;    try {
;       may_throw();
;    } catch (int i) {
;      try {
;        may_throw();
;      }
;      catch (int j) {
;        i = j;
;      }
;      handle_int(i);
;    }
;  } catch (float f) {
;    handle_float(f);
;  }
;  done();
;}

; ModuleID = 'cppeh-nested-3.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }

$"\01??_R0M@8" = comdat any

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0M@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".M\00" }, comdat
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat

; CHECK: define void @"\01?test@@YAXXZ"()
; CHECK: entry:
; CHECK:   %i = alloca i32, align 4
; CHECK:   %j = alloca i32, align 4
; CHECK:   %f = alloca float, align 4
; CHECK:   call void (...) @llvm.localescape(i32* %j, i32* %i, float* %f)
; CHECK:   invoke void @"\01?may_throw@@YAXXZ"()
; CHECK:           to label %invoke.cont unwind label %[[LPAD_LABEL:lpad[0-9]*]]

; Function Attrs: uwtable
define void @"\01?test@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %f = alloca float, align 4
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  br label %try.cont10

; CHECK: [[LPAD_LABEL]]:
; CHECK:   landingpad { i8*, i32 }
; CHECK:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
; CHECK:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0M@8" to i8*)
; CHECK:   [[RECOVER:\%.+]] = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*), i32 1, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch.2", i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0M@8" to i8*), i32 2, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch.1")
; CHECK:   indirectbr i8* [[RECOVER]], [label %try.cont10, label %try.cont19]

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0M@8" to i8*)
  %1 = extractvalue { i8*, i32 } %0, 0
  store i8* %1, i8** %exn.slot
  %2 = extractvalue { i8*, i32 } %0, 1
  store i32 %2, i32* %ehselector.slot
  br label %catch.dispatch

; CHECK-NOT: catch.dispatch:
catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, i32* %ehselector.slot
  %3 = call i32 @llvm.eh.typeid.for(i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)) #3
  %matches = icmp eq i32 %sel, %3
  br i1 %matches, label %catch, label %catch.dispatch11

; CHECK-NOT: catch:
catch:                                            ; preds = %catch.dispatch
  %exn = load i8*, i8** %exn.slot
  %4 = bitcast i32* %i to i8*
  call void @llvm.eh.begincatch(i8* %exn, i8* %4) #3
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont2 unwind label %lpad1

; CHECK-NOT: invoke.cont2:
invoke.cont2:                                     ; preds = %catch
  br label %try.cont

; CHECK-NOT: lpad1:
lpad1:                                            ; preds = %catch
  %5 = landingpad { i8*, i32 }
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0M@8" to i8*)
  %6 = extractvalue { i8*, i32 } %5, 0
  store i8* %6, i8** %exn.slot
  %7 = extractvalue { i8*, i32 } %5, 1
  store i32 %7, i32* %ehselector.slot
  br label %catch.dispatch3

; CHECK-NOT: catch.dispatch3:
catch.dispatch3:                                  ; preds = %lpad1
  %sel4 = load i32, i32* %ehselector.slot
  %8 = call i32 @llvm.eh.typeid.for(i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)) #3
  %matches5 = icmp eq i32 %sel4, %8
  br i1 %matches5, label %catch6, label %catch.dispatch11

; CHECK-NOT: catch6:
catch6:                                           ; preds = %catch.dispatch3
  %exn7 = load i8*, i8** %exn.slot
  %9 = bitcast i32* %j to i8*
  call void @llvm.eh.begincatch(i8* %exn7, i8* %9) #3
  %10 = load i32, i32* %j, align 4
  store i32 %10, i32* %i, align 4
  call void @llvm.eh.endcatch() #3
  br label %try.cont

; CHECK-NOT: try.cont:
try.cont:                                         ; preds = %catch6, %invoke.cont2
  %11 = load i32, i32* %i, align 4
  invoke void @"\01?handle_int@@YAXH@Z"(i32 %11)
          to label %invoke.cont9 unwind label %lpad8

; CHECK-NOT: invoke.cont9:
invoke.cont9:                                     ; preds = %try.cont
  call void @llvm.eh.endcatch() #3
  br label %try.cont10

try.cont10:                                       ; preds = %invoke.cont9, %invoke.cont
  br label %try.cont19

; CHECK-NOT: lpad8:
lpad8:                                            ; preds = %try.cont
  %12 = landingpad { i8*, i32 }
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0M@8" to i8*)
  %13 = extractvalue { i8*, i32 } %12, 0
  store i8* %13, i8** %exn.slot
  %14 = extractvalue { i8*, i32 } %12, 1
  store i32 %14, i32* %ehselector.slot
  call void @llvm.eh.endcatch() #3
  br label %catch.dispatch11

; CHECK-NOT: catch.dispatch11:
catch.dispatch11:                                 ; preds = %lpad8, %catch.dispatch3, %catch.dispatch
  %sel12 = load i32, i32* %ehselector.slot
  %15 = call i32 @llvm.eh.typeid.for(i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0M@8" to i8*)) #3
  %matches13 = icmp eq i32 %sel12, %15
  br i1 %matches13, label %catch14, label %eh.resume

; CHECK-NOT: catch14:
catch14:                                          ; preds = %catch.dispatch11
  %exn15 = load i8*, i8** %exn.slot
  %16 = bitcast float* %f to i8*
  call void @llvm.eh.begincatch(i8* %exn15, i8* %16) #3
  %17 = load float, float* %f, align 4
  call void @"\01?handle_float@@YAXM@Z"(float %17)
  call void @llvm.eh.endcatch() #3
  br label %try.cont19

try.cont19:                                       ; preds = %catch14, %try.cont10
  call void @"\01?done@@YAXXZ"()
  ret void

; CHECK-NOT: eh.resume:
eh.resume:                                        ; preds = %lpad16, %catch.dispatch11
  %exn20 = load i8*, i8** %exn.slot
  %sel21 = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn20, 0
  %lpad.val22 = insertvalue { i8*, i32 } %lpad.val, i32 %sel21, 1
  resume { i8*, i32 } %lpad.val22
; CHECK: }
}

; CHECK: define internal i8* @"\01?test@@YAXXZ.catch"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_J:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 0)
; CHECK:   [[J_PTR:\%.+]] = bitcast i8* [[RECOVER_J]] to i32*
; CHECK:   [[RECOVER_I1:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 1)
; CHECK:   [[I_PTR1:\%.+]] = bitcast i8* [[RECOVER_I1]] to i32*
; CHECK:   [[TMP3:\%.+]] = load i32, i32* [[J_PTR]], align 4
; CHECK:   store i32 [[TMP3]], i32* [[I_PTR1]]
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ.catch.2", %invoke.cont2)
; CHECK: }

; CHECK: define internal i8* @"\01?test@@YAXXZ.catch.1"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_F:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 2)
; CHECK:   [[F_PTR:\%.+]] = bitcast i8* [[RECOVER_F]] to float*
; CHECK:   [[TMP2:\%.+]] = load float, float* [[F_PTR]], align 4
; CHECK:   call void @"\01?handle_float@@YAXM@Z"(float [[TMP2]])
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont19)
; CHECK: }

; CHECK: define internal i8* @"\01?test@@YAXXZ.catch.2"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_I:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 1)
; CHECK:   [[I_PTR:\%.+]] = bitcast i8* [[RECOVER_I]] to i32*
; CHECK:   invoke void @"\01?may_throw@@YAXXZ"()
; CHECK:           to label %invoke.cont2 unwind label %[[LPAD1_LABEL:lpad[0-9]*]]
;
; CHECK: invoke.cont2:                                     ; preds = %[[LPAD1_LABEL]], %entry
; CHECK:   [[TMP1:\%.+]] = load i32, i32* [[I_PTR]], align 4
; CHECK:   invoke void @"\01?handle_int@@YAXH@Z"(i32 [[TMP1]])
; CHECK:           to label %invoke.cont9 unwind label %[[LPAD8_LABEL:lpad[0-9]*]]
;
; CHECK: [[LPAD1_LABEL]]:{{[ ]+}}; preds = %entry
; CHECK:   [[LPAD1_VAL:\%.+]] = landingpad { i8*, i32 }
; CHECK:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
; CHECK:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0M@8" to i8*)
; CHECK:   [[RECOVER1:\%.+]] = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*), i32 0, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch", i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0M@8" to i8*), i32 2, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch.1")
; CHECK:   indirectbr i8* [[RECOVER1]], [label %invoke.cont2]
;
; CHECK: invoke.cont9:
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont10)
;
; CHECK: [[LPAD8_LABEL]]:{{[ ]+}}; preds = %invoke.cont2
; CHECK:   [[LPAD8_VAL:\%.+]] = landingpad { i8*, i32 }
; CHECK:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0M@8" to i8*)
; CHECK:   [[RECOVER2:\%.+]] = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0M@8" to i8*), i32 2, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch.1")
; CHECK:   indirectbr i8* [[RECOVER2]], []
;
; CHECK: }

declare void @"\01?may_throw@@YAXXZ"() #1

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #3

declare void @"\01?handle_int@@YAXH@Z"(i32) #1

declare void @"\01?handle_float@@YAXM@Z"(float) #1

declare void @"\01?done@@YAXXZ"() #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (trunk 232069) (llvm/trunk 232070)"}
