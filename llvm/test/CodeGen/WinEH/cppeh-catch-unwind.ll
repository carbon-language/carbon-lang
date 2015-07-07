; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test was generated from the following source:
;
; void test() {
;   try {
;     SomeClass obj;
;     may_throw();
;     try {
;       may_throw();
;     } catch (int) {
;       handle_exception();
;     }
;   } catch (int) {
;     handle_exception();
;   }
; }
;
; The code above was compiled with the -O2 option.

; ModuleID = 'catch-unwind.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%class.SomeClass = type { i8 }

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat


; CHECK-LABEL: define void @"\01?test@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
; CHECK: entry:
; CHECK:   [[OBJ_PTR:\%.+]] = alloca %class.SomeClass
; CHECK:   [[TMP0:\%.+]] = alloca i32, align 4
; CHECK:   [[TMP1:\%.+]] = alloca i32, align 4
; CHECK:   call void (...) @llvm.localescape(i32* [[TMP1]], %class.SomeClass* [[OBJ_PTR]], i32* [[TMP0]])
; CHECK:   %call = invoke %class.SomeClass* @"\01??0SomeClass@@QEAA@XZ"(%class.SomeClass* %obj)
; CHECK:           to label %invoke.cont unwind label %[[LPAD_LABEL:lpad[0-9]*]]

; Function Attrs: uwtable
define void @"\01?test@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %obj = alloca %class.SomeClass, align 1
  %0 = alloca i32, align 4
  %1 = alloca i32, align 4
  %call = invoke %class.SomeClass* @"\01??0SomeClass@@QEAA@XZ"(%class.SomeClass* %obj)
          to label %invoke.cont unwind label %lpad

; CHECK: invoke.cont:
; CHECK:   invoke void @"\01?may_throw@@YAXXZ"()
; CHECK:           to label %invoke.cont2 unwind label %[[LPAD1_LABEL:lpad[0-9]*]]

invoke.cont:                                      ; preds = %entry
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont2 unwind label %lpad1

; CHECK: invoke.cont2:
; CHECK:   invoke void @"\01?may_throw@@YAXXZ"()
; CHECK:           to label %try.cont unwind label %[[LPAD3_LABEL:lpad[0-9]*]]

invoke.cont2:                                     ; preds = %invoke.cont
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %try.cont unwind label %lpad3

; CHECK: [[LPAD_LABEL]]:{{[ ]+}}; preds = %entry
; CHECK:   [[LPAD_VAL:\%.+]] = landingpad { i8*, i32 }
; CHECK-NEXT:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
; CHECK-NEXT:   [[RECOVER:\%.+]] = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*), i32 0, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch")
; CHECK-NEXT:   indirectbr i8* [[RECOVER]], [label %try.cont15]

lpad:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 }
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
  %3 = extractvalue { i8*, i32 } %2, 0
  %4 = extractvalue { i8*, i32 } %2, 1
  br label %catch.dispatch7

; CHECK: [[LPAD1_LABEL]]:{{[ ]+}}; preds = %invoke.cont
; CHECK:   [[LPAD1_VAL:\%.+]] = landingpad { i8*, i32 }
; CHECK-NEXT:           cleanup
; CHECK-NEXT:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
; CHECK-NEXT:   [[RECOVER1:\%.+]] = call i8* (...) @llvm.eh.actions(i32 0, void (i8*, i8*)* @"\01?test@@YAXXZ.cleanup", i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*), i32 0, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch")
; CHECK-NEXT:   indirectbr i8* [[RECOVER1]], [label %try.cont15]

lpad1:                                            ; preds = %invoke.cont
  %5 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
  %6 = extractvalue { i8*, i32 } %5, 0
  %7 = extractvalue { i8*, i32 } %5, 1
  br label %ehcleanup

; CHECK: [[LPAD3_LABEL]]:{{[ ]+}}; preds = %invoke.cont2
; CHECK:   [[LPAD3_VAL:\%.+]] = landingpad { i8*, i32 }
; CHECK-NEXT:           cleanup
; CHECK-NEXT:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
; CHECK-NEXT:   [[RECOVER3:\%.+]] = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*), i32 2, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch.1", i32 0, void (i8*, i8*)* @"\01?test@@YAXXZ.cleanup")
; CHECK-NEXT:   indirectbr i8* [[RECOVER3]], [label %try.cont, label %try.cont15]

lpad3:                                            ; preds = %invoke.cont2
  %8 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
  %9 = extractvalue { i8*, i32 } %8, 0
  %10 = extractvalue { i8*, i32 } %8, 1
  %11 = call i32 @llvm.eh.typeid.for(i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)) #3
  %matches = icmp eq i32 %10, %11
  br i1 %matches, label %catch, label %ehcleanup

; CHECK-NOT: catch:
catch:                                            ; preds = %lpad3
  %12 = bitcast i32* %0 to i8*
  call void @llvm.eh.begincatch(i8* %9, i8* %12) #3
  invoke void @"\01?handle_exception@@YAXXZ"()
          to label %invoke.cont6 unwind label %lpad5

; CHECK-NOT: invoke.cont6:
invoke.cont6:                                     ; preds = %catch
  call void @llvm.eh.endcatch() #3
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %invoke.cont6
  call void @"\01??1SomeClass@@QEAA@XZ"(%class.SomeClass* %obj) #3
  br label %try.cont15

; CHECK-NOT: lpad5:
lpad5:                                            ; preds = %catch
  %13 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
  %14 = extractvalue { i8*, i32 } %13, 0
  %15 = extractvalue { i8*, i32 } %13, 1
  call void @llvm.eh.endcatch() #3
  br label %ehcleanup

; CHECK-NOT: ehcleanup
ehcleanup:                                        ; preds = %lpad5, %lpad3, %lpad1
  %exn.slot.0 = phi i8* [ %14, %lpad5 ], [ %9, %lpad3 ], [ %6, %lpad1 ]
  %ehselector.slot.0 = phi i32 [ %15, %lpad5 ], [ %10, %lpad3 ], [ %7, %lpad1 ]
  call void @"\01??1SomeClass@@QEAA@XZ"(%class.SomeClass* %obj) #3
  br label %catch.dispatch7

; CHECK-NOT: catch.dispatch7:
catch.dispatch7:                                  ; preds = %ehcleanup, %lpad
  %exn.slot.1 = phi i8* [ %exn.slot.0, %ehcleanup ], [ %3, %lpad ]
  %ehselector.slot.1 = phi i32 [ %ehselector.slot.0, %ehcleanup ], [ %4, %lpad ]
  %16 = call i32 @llvm.eh.typeid.for(i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)) #3
  %matches9 = icmp eq i32 %ehselector.slot.1, %16
  br i1 %matches9, label %catch10, label %eh.resume

; CHECK-NOT: catch10:
catch10:                                          ; preds = %catch.dispatch7
  %17 = bitcast i32* %1 to i8*
  call void @llvm.eh.begincatch(i8* %exn.slot.1, i8* %17) #3
  call void @"\01?handle_exception@@YAXXZ"()
  br label %invoke.cont13

; CHECK-NOT: invoke.cont13:
invoke.cont13:                                    ; preds = %catch10
  call void @llvm.eh.endcatch() #3
  br label %try.cont15

try.cont15:                                       ; preds = %invoke.cont13, %try.cont
  ret void

; CHECK-NOT: eh.resume
eh.resume:                                        ; preds = %catch.dispatch7
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.1, 0
  %lpad.val18 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.1, 1
  resume { i8*, i32 } %lpad.val18

; CHECK: }
}

; CHECK-LABEL: define internal i8* @"\01?test@@YAXXZ.catch"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_TMP1:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 0)
; CHECK:   [[TMP1_PTR:\%.+]] = bitcast i8* [[RECOVER_TMP1]] to i32*
; CHECK:   call void @"\01?handle_exception@@YAXXZ"()
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont15)
; CHECK: }

; CHECK-LABEL: define internal void @"\01?test@@YAXXZ.cleanup"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_OBJ:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 1)
; CHECK:   [[OBJ_PTR:\%.+]] = bitcast i8* %obj.i8 to %class.SomeClass*
; CHECK:   call void @"\01??1SomeClass@@QEAA@XZ"(%class.SomeClass* [[OBJ_PTR]])
; CHECK:   ret void
; CHECK: }

; CHECK-LABEL: define internal i8* @"\01?test@@YAXXZ.catch.1"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_TMP0:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 2)
; CHECK:   [[TMP0_PTR:\%.+]] = bitcast i8* [[RECOVER_TMP0]] to i32*
; CHECK:   invoke void @"\01?handle_exception@@YAXXZ"()
; CHECK:           to label %invoke.cont6 unwind label %[[LPAD5_LABEL:lpad[0-9]+]]
;
; CHECK: invoke.cont6:                                     ; preds = %entry
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont)
;
; CHECK: [[LPAD5_LABEL]]:{{[ ]+}}; preds = %entry
; CHECK:   [[LPAD5_VAL:\%.+]] = landingpad { i8*, i32 }
; CHECK:           cleanup
; CHECK:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
; CHECK: }

declare %class.SomeClass* @"\01??0SomeClass@@QEAA@XZ"(%class.SomeClass* returned) #1

declare i32 @__CxxFrameHandler3(...)

declare void @"\01?may_throw@@YAXXZ"() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #3

declare void @"\01?handle_exception@@YAXXZ"() #1

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #3

; Function Attrs: nounwind
declare void @"\01??1SomeClass@@QEAA@XZ"(%class.SomeClass*) #4

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }
attributes #4 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (trunk 232069) (llvm/trunk 232070)"}
