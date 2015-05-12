; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following code:
;
; void test()
; {
;   try {
;     may_throw();
;   } catch (int i) {
;     handle_int(i);
;   } catch (long long ll) {
;     handle_long_long(ll);
;   } catch (SomeClass &obj) {
;     handle_obj(&obj);
;   } catch (...) {
;     handle_exception();
;   }
; }
;
; The catch handlers were edited to insert 'ret void' after the endcatch call.

; ModuleID = 'catch-with-type.cpp'
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.HandlerMapEntry = type { i32, i32 }
%rtti.TypeDescriptor3 = type { i8**, i8*, [4 x i8] }
%rtti.TypeDescriptor15 = type { i8**, i8*, [16 x i8] }
%class.SomeClass = type { i8 }

$"\01??_R0H@8" = comdat any

$"\01??_R0_J@8" = comdat any

$"\01??_R0?AVSomeClass@@@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@__ImageBase = external constant i8
@llvm.eh.handlermapentry.H = private unnamed_addr constant %eh.HandlerMapEntry { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section "llvm.metadata"
@"\01??_R0_J@8" = linkonce_odr global %rtti.TypeDescriptor3 { i8** @"\01??_7type_info@@6B@", i8* null, [4 x i8] c"._J\00" }, comdat
@llvm.eh.handlermapentry._J = private unnamed_addr constant %eh.HandlerMapEntry { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor3* @"\01??_R0_J@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section "llvm.metadata"
@"\01??_R0?AVSomeClass@@@8" = linkonce_odr global %rtti.TypeDescriptor15 { i8** @"\01??_7type_info@@6B@", i8* null, [16 x i8] c".?AVSomeClass@@\00" }, comdat
@"llvm.eh.handlermapentry.reference.?AVSomeClass@@" = private unnamed_addr constant %eh.HandlerMapEntry { i32 8, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor15* @"\01??_R0?AVSomeClass@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section "llvm.metadata"


; CHECK: define void @"\01?test@@YAXXZ"() #0 {
; CHECK: entry:
; CHECK:   [[OBJ_PTR:\%.+]] = alloca %class.SomeClass*, align 8
; CHECK:   [[LL_PTR:\%.+]] = alloca i64, align 8
; CHECK:   [[I_PTR:\%.+]] = alloca i32, align 4
; CHECK:   call void (...) @llvm.frameescape(i32* [[I_PTR]], i64* [[LL_PTR]], %class.SomeClass** [[OBJ_PTR]])
; CHECK:   invoke void @"\01?may_throw@@YAXXZ"()
; CHECK:           to label %invoke.cont unwind label %[[LPAD_LABEL:lpad[0-9]*]]

; Function Attrs: uwtable
define void @"\01?test@@YAXXZ"() #0 {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %obj = alloca %class.SomeClass*, align 8
  %ll = alloca i64, align 8
  %i = alloca i32, align 4
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  br label %try.cont

; CHECK: [[LPAD_LABEL]]:{{[ ]+}}; preds = %entry
; CHECK:   landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
; CHECK-NEXT:           catch %eh.HandlerMapEntry* @llvm.eh.handlermapentry.H
; CHECK-NEXT:           catch %eh.HandlerMapEntry* @llvm.eh.handlermapentry._J
; CHECK-NEXT:           catch %eh.HandlerMapEntry* @"llvm.eh.handlermapentry.reference.?AVSomeClass@@"
; CHECK-NEXT:           catch i8* null
; CHECK-NEXT:   [[RECOVER:\%.+]] = call i8* (...) @llvm.eh.actions(
; CHECK-SAME:	     i32 1, i8* bitcast (%eh.HandlerMapEntry* @llvm.eh.handlermapentry.H to i8*), i32 0, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch",
; CHECK-SAME:        i32 1, i8* bitcast (%eh.HandlerMapEntry* @llvm.eh.handlermapentry._J to i8*), i32 1, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch.1",
; CHECK-SAME:        i32 1, i8* bitcast (%eh.HandlerMapEntry* @"llvm.eh.handlermapentry.reference.?AVSomeClass@@" to i8*), i32 2, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch.2",
; CHECK-SAME:        i32 1, i8* null, i32 -1, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch.3")
; CHECK-NEXT:   indirectbr i8* [[RECOVER]], [label %ret]

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch %eh.HandlerMapEntry* @llvm.eh.handlermapentry.H
          catch %eh.HandlerMapEntry* @llvm.eh.handlermapentry._J
          catch %eh.HandlerMapEntry* @"llvm.eh.handlermapentry.reference.?AVSomeClass@@"
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  store i8* %1, i8** %exn.slot
  %2 = extractvalue { i8*, i32 } %0, 1
  store i32 %2, i32* %ehselector.slot
  br label %catch.dispatch

; CHECK-NOT: catch.dispatch:
catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, i32* %ehselector.slot
  %3 = call i32 @llvm.eh.typeid.for(i8* bitcast (%eh.HandlerMapEntry* @llvm.eh.handlermapentry.H to i8*)) #3
  %matches = icmp eq i32 %sel, %3
  br i1 %matches, label %catch14, label %catch.fallthrough

ret:
  ret void

; CHECK-NOT: catch14:
; CHECK: ret:
; CHECK-NEXT:   ret void
catch14:                                          ; preds = %catch.dispatch
  %exn15 = load i8*, i8** %exn.slot
  %4 = bitcast i32* %i to i8*
  call void @llvm.eh.begincatch(i8* %exn15, i8* %4) #3
  %5 = load i32, i32* %i, align 4
  call void @"\01?handle_int@@YAXH@Z"(i32 %5)
  call void @llvm.eh.endcatch() #3
  br label %ret

try.cont:                                         ; preds = %invoke.cont
  br label %ret

; CHECK-NOT: catch.fallthrough:
catch.fallthrough:                                ; preds = %catch.dispatch
  %6 = call i32 @llvm.eh.typeid.for(i8* bitcast (%eh.HandlerMapEntry* @llvm.eh.handlermapentry._J to i8*)) #3
  %matches1 = icmp eq i32 %sel, %6
  br i1 %matches1, label %catch10, label %catch.fallthrough2

; CHECK-NOT: catch10:
catch10:                                          ; preds = %catch.fallthrough
  %exn11 = load i8*, i8** %exn.slot
  %7 = bitcast i64* %ll to i8*
  call void @llvm.eh.begincatch(i8* %exn11, i8* %7) #3
  %8 = load i64, i64* %ll, align 8
  call void @"\01?handle_long_long@@YAX_J@Z"(i64 %8)
  call void @llvm.eh.endcatch() #3
  br label %ret

; CHECK-NOT: catch.fallthrough2:
catch.fallthrough2:                               ; preds = %catch.fallthrough
  %9 = call i32 @llvm.eh.typeid.for(i8* bitcast (%eh.HandlerMapEntry* @"llvm.eh.handlermapentry.reference.?AVSomeClass@@" to i8*)) #3
  %matches3 = icmp eq i32 %sel, %9
  br i1 %matches3, label %catch6, label %catch

; CHECK-NOT: catch6:
catch6:                                           ; preds = %catch.fallthrough2
  %exn7 = load i8*, i8** %exn.slot
  %10 = bitcast %class.SomeClass** %obj to i8*
  call void @llvm.eh.begincatch(i8* %exn7, i8* %10) #3
  %11 = load %class.SomeClass*, %class.SomeClass** %obj, align 8
  call void @"\01?handle_obj@@YAXPEAVSomeClass@@@Z"(%class.SomeClass* %11)
  call void @llvm.eh.endcatch() #3
  br label %ret

; CHECK-NOT: catch:
catch:                                            ; preds = %catch.fallthrough2
  %exn = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn, i8* null) #3
  call void @"\01?handle_exception@@YAXXZ"()  call void @llvm.eh.endcatch() #3
  br label %ret
; CHECK: }
}

; CHECK-LABEL: define internal i8* @"\01?test@@YAXXZ.catch"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_I:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 0)
; CHECK:   [[I_PTR:\%.+]] = bitcast i8* [[RECOVER_I]] to i32*
; CHECK:   [[TMP1:\%.+]] = load i32, i32* [[I_PTR]], align 4
; CHECK:   call void @"\01?handle_int@@YAXH@Z"(i32 [[TMP1]])
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %ret)
; CHECK: }

; CHECK-LABEL: define internal i8* @"\01?test@@YAXXZ.catch.1"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_LL:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 1)
; CHECK:   [[LL_PTR:\%.+]] = bitcast i8* [[RECOVER_LL]] to i64*
; CHECK:   [[TMP2:\%.+]] = load i64, i64* [[LL_PTR]], align 8
; CHECK:   call void @"\01?handle_long_long@@YAX_J@Z"(i64 [[TMP2]])
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %ret)
; CHECK: }

; CHECK-LABEL: define internal i8* @"\01?test@@YAXXZ.catch.2"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_OBJ:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 2)
; CHECK:   [[OBJ_PTR:\%.+]] = bitcast i8* [[RECOVER_OBJ]] to %class.SomeClass**
; CHECK:   [[TMP3:\%.+]] = load %class.SomeClass*, %class.SomeClass** [[OBJ_PTR]], align 8
; CHECK:   call void @"\01?handle_obj@@YAXPEAVSomeClass@@@Z"(%class.SomeClass* [[TMP3]])
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %ret)
; CHECK: }

; CHECK-LABEL: define internal i8* @"\01?test@@YAXXZ.catch.3"(i8*, i8*)
; CHECK: entry:
; CHECK:   call void @"\01?handle_exception@@YAXXZ"()
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %ret)
; CHECK: }


declare void @"\01?may_throw@@YAXXZ"() #1

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #3

declare void @"\01?handle_exception@@YAXXZ"() #1

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #3

declare void @"\01?handle_obj@@YAXPEAVSomeClass@@@Z"(%class.SomeClass*) #1

declare void @"\01?handle_long_long@@YAX_J@Z"(i64) #1

declare void @"\01?handle_int@@YAXH@Z"(i32) #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (trunk 233155) (llvm/trunk 233153)"}
