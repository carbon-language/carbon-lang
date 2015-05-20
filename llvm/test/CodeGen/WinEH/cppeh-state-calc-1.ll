; RUN: llc < %s | FileCheck %s

; This test was generated from the following code.
;
; void test() {
;   try {
;     try {
;       try {
;         two();
;         throw 2;
;       } catch (int x) {
;         catch_two();
;       }
;       a();
;       throw 'a';
;     } catch (char c) {
;       catch_a();
;     }
;     one();
;     throw 1;
;   } catch(int x) { 
;     catch_one();
;   } catch(...) {
;     catch_all();
;   }
; }
;
; The function calls before the throws were declared as 'noexcept' and are
; just here to make blocks easier to identify in the IR.

; ModuleID = '<stdin>'
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchHandlerType = type { i32, i8* }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }

$"\01??_R0H@8" = comdat any

$"\01??_R0D@8" = comdat any

$"_CT??_R0H@84" = comdat any

$_CTA1H = comdat any

$_TI1H = comdat any

$"_CT??_R0D@81" = comdat any

$_CTA1D = comdat any

$_TI1D = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@llvm.eh.handlertype.H.0 = private unnamed_addr constant %eh.CatchHandlerType { i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*) }, section "llvm.metadata"
@"\01??_R0D@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".D\00" }, comdat
@llvm.eh.handlertype.D.0 = private unnamed_addr constant %eh.CatchHandlerType { i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0D@8" to i8*) }, section "llvm.metadata"
@__ImageBase = external constant i8
@"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
@_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableType* @"_CT??_R0H@84" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableTypeArray.1* @_CTA1H to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"_CT??_R0D@81" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor2* @"\01??_R0D@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 1, i32 0 }, section ".xdata", comdat
@_CTA1D = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableType* @"_CT??_R0D@81" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@_TI1D = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableTypeArray.1* @_CTA1D to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section ".xdata", comdat

; Function Attrs: nounwind uwtable
define void @"\01?test@@YAXXZ"() #0 {
entry:
  %tmp = alloca i32, align 4
  %x = alloca i32, align 4
  %tmp2 = alloca i8, align 1
  %c = alloca i8, align 1
  %tmp11 = alloca i32, align 4
  %x21 = alloca i32, align 4
  call void @"\01?two@@YAXXZ"() #3
  store i32 2, i32* %tmp
  %0 = bitcast i32* %tmp to i8*
  call void (...) @llvm.frameescape(i32* %x, i8* %c, i32* %x21)
  invoke void @_CxxThrowException(i8* %0, %eh.ThrowInfo* @_TI1H) #5
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*)
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.D.0
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.0
          catch i8* null
  %recover = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*), i32 0, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch", i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.D.0 to i8*), i32 1, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch1", i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*), i32 2, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch2", i32 1, i8* null, i32 -1, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch3")
  indirectbr i8* %recover, [label %try.cont, label %try.cont10, label %try.cont22]

try.cont:                                         ; preds = %lpad
  call void @"\01?a@@YAXXZ"() #3
  store i8 97, i8* %tmp2
  invoke void @_CxxThrowException(i8* %tmp2, %eh.ThrowInfo* @_TI1D) #5
          to label %unreachable unwind label %lpad3

lpad3:                                            ; preds = %try.cont
  %2 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.D.0
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.0
          catch i8* null
  %recover1 = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.D.0 to i8*), i32 1, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch1", i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*), i32 2, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch2", i32 1, i8* null, i32 -1, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch3")
  indirectbr i8* %recover1, [label %try.cont10, label %try.cont22]

try.cont10:                                       ; preds = %lpad3, %lpad
  call void @"\01?one@@YAXXZ"() #3
  store i32 1, i32* %tmp11
  %3 = bitcast i32* %tmp11 to i8*
  invoke void @_CxxThrowException(i8* %3, %eh.ThrowInfo* @_TI1H) #5
          to label %unreachable unwind label %lpad12

lpad12:                                           ; preds = %try.cont10
  %4 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.0
          catch i8* null
  %recover2 = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*), i32 2, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch2", i32 1, i8* null, i32 -1, i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch3")
  indirectbr i8* %recover2, [label %try.cont22]

try.cont22:                                       ; preds = %lpad12, %lpad3, %lpad
  ret void

unreachable:                                      ; preds = %try.cont10, %try.cont, %entry
  unreachable
}

; Function Attrs: nounwind
declare void @"\01?two@@YAXXZ"() #1

declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #3

; Function Attrs: nounwind
declare void @"\01?catch_two@@YAXXZ"() #1

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #3

; Function Attrs: nounwind
declare void @"\01?a@@YAXXZ"() #1

; Function Attrs: nounwind
declare void @"\01?catch_a@@YAXXZ"() #1

; Function Attrs: nounwind
declare void @"\01?one@@YAXXZ"() #1

; Function Attrs: nounwind
declare void @"\01?catch_all@@YAXXZ"() #1

; Function Attrs: nounwind
declare void @"\01?catch_one@@YAXXZ"() #1

; Function Attrs: nounwind
declare i8* @llvm.eh.actions(...) #3

define internal i8* @"\01?test@@YAXXZ.catch"(i8*, i8*) #4 {
entry:
  %x.i8 = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 0)
  %x = bitcast i8* %x.i8 to i32*
  %2 = bitcast i32* %x to i8*
  call void @"\01?catch_two@@YAXXZ"() #3
  invoke void @llvm.donothing()
          to label %entry.split unwind label %stub

entry.split:                                      ; preds = %entry
  ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont)

stub:                                             ; preds = %entry
  %3 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
  %recover = call i8* (...) @llvm.eh.actions()
  unreachable
}

; Function Attrs: nounwind readnone
declare void @llvm.donothing() #2

define internal i8* @"\01?test@@YAXXZ.catch1"(i8*, i8*) #4 {
entry:
  call void @"\01?catch_a@@YAXXZ"() #3
  invoke void @llvm.donothing()
          to label %entry.split unwind label %stub

entry.split:                                      ; preds = %entry
  ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont10)

stub:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
  %recover = call i8* (...) @llvm.eh.actions()
  unreachable
}

define internal i8* @"\01?test@@YAXXZ.catch2"(i8*, i8*) #4 {
entry:
  %x21.i8 = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 2)
  %x21 = bitcast i8* %x21.i8 to i32*
  %2 = bitcast i32* %x21 to i8*
  call void @"\01?catch_one@@YAXXZ"() #3
  invoke void @llvm.donothing()
          to label %entry.split unwind label %stub

entry.split:                                      ; preds = %entry
  ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont22)

stub:                                             ; preds = %entry
  %3 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
  %recover = call i8* (...) @llvm.eh.actions()
  unreachable
}

define internal i8* @"\01?test@@YAXXZ.catch3"(i8*, i8*) #4 {
entry:
  call void @"\01?catch_all@@YAXXZ"() #3
  invoke void @llvm.donothing()
          to label %entry.split unwind label %stub

entry.split:                                      ; preds = %entry
  ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont22)

stub:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
  %recover = call i8* (...) @llvm.eh.actions()
  unreachable
}

; Function Attrs: nounwind
declare void @llvm.frameescape(...) #3

; Function Attrs: nounwind readnone
declare i8* @llvm.framerecover(i8*, i8*, i32) #2

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" "wineh-parent"="?test@@YAXXZ" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }
attributes #4 = { "wineh-parent"="?test@@YAXXZ" }
attributes #5 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (trunk 236059)"}

; CHECK-LABEL: "$cppxdata$?test@@YAXXZ":
; CHECK-NEXT: 	.long	429065506
; CHECK-NEXT: 	.long
; CHECK-NEXT: 	.long	("$stateUnwindMap$?test@@YAXXZ")@IMGREL
; CHECK-NEXT: 	.long
; CHECK-NEXT: 	.long	("$tryMap$?test@@YAXXZ")@IMGREL
; CHECK-NEXT: 	.long
; CHECK-NEXT: 	.long	("$ip2state$?test@@YAXXZ")@IMGREL
; CHECK-NEXT: 	.long	40
; CHECK-NEXT: 	.long	0
; CHECK-NEXT: 	.long	1
; CHECK: "$stateUnwindMap$?test@@YAXXZ":
; CHECK: "$tryMap$?test@@YAXXZ":
; CHECK: "$handlerMap$0$?test@@YAXXZ":
; CHECK: "$ip2state$?test@@YAXXZ":
; CHECK-NEXT: 	.long	.Lfunc_begin0@IMGREL
; CHECK-NEXT: 	.long	-1
; CHECK-NEXT: 	.long	.Ltmp0@IMGREL
; CHECK-NEXT: 	.long	2
; CHECK-NEXT: 	.long	.Ltmp3@IMGREL
; CHECK-NEXT: 	.long	1
; CHECK-NEXT: 	.long	.Ltmp6@IMGREL
; CHECK-NEXT: 	.long	0
; CHECK-NEXT: 	.long	.Lfunc_begin1@IMGREL
; CHECK-NEXT: 	.long	3
; CHECK-NEXT: 	.long	.Lfunc_begin2@IMGREL
; CHECK-NEXT: 	.long	4
; CHECK-NEXT: 	.long	.Lfunc_begin3@IMGREL
; CHECK-NEXT: 	.long	5
; CHECK-NEXT: 	.long	.Lfunc_begin4@IMGREL
; CHECK-NEXT: 	.long	6
