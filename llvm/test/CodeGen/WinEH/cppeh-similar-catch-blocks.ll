; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following code:
;
; int main(void) {
;   try {
;     try {
;       throw 'a';
;     } catch (char c) {
;       printf("%c\n", c);
;     }
;     throw 1;
;   } catch(int x) {
;     printf("%d\n", x);
;   } catch(...) {
;     printf("...\n");
;   }
;   try {
;     try {
;       throw 'b';
;     } catch (char c) {
;       printf("%c\n", c);
;     }
;     throw 2;
;   } catch(int x) {
;     printf("%d\n", x);
;   } catch (char c) {
;     printf("%c\n", c);
;   } catch(...) {
;     printf("...\n");
;   }
;   return 0;
; }

; This test is just checking for failures in processing the IR.
; Extensive handler matching is not required.

; ModuleID = 'cppeh-similar-catch-blocks.cpp'
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchHandlerType = type { i32, i8* }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }

$"\01??_R0H@8" = comdat any

$"\01??_R0D@8" = comdat any

$"_CT??_R0D@81" = comdat any

$_CTA1D = comdat any

$_TI1D = comdat any

$"\01??_C@_03PJCJOCBM@?$CFc?6?$AA@" = comdat any

$"_CT??_R0H@84" = comdat any

$_CTA1H = comdat any

$_TI1H = comdat any

$"\01??_C@_04MPPNMCOK@?4?4?4?6?$AA@" = comdat any

$"\01??_C@_03PMGGPEJJ@?$CFd?6?$AA@" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@llvm.eh.handlertype.H.0 = private unnamed_addr constant %eh.CatchHandlerType { i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*) }, section "llvm.metadata"
@"\01??_R0D@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".D\00" }, comdat
@llvm.eh.handlertype.D.0 = private unnamed_addr constant %eh.CatchHandlerType { i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0D@8" to i8*) }, section "llvm.metadata"
@__ImageBase = external constant i8
@"_CT??_R0D@81" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor2* @"\01??_R0D@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 1, i32 0 }, section ".xdata", comdat
@_CTA1D = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableType* @"_CT??_R0D@81" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@_TI1D = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableTypeArray.1* @_CTA1D to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"\01??_C@_03PJCJOCBM@?$CFc?6?$AA@" = linkonce_odr unnamed_addr constant [4 x i8] c"%c\0A\00", comdat, align 1
@"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
@_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableType* @"_CT??_R0H@84" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableTypeArray.1* @_CTA1H to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"\01??_C@_04MPPNMCOK@?4?4?4?6?$AA@" = linkonce_odr unnamed_addr constant [5 x i8] c"...\0A\00", comdat, align 1
@"\01??_C@_03PMGGPEJJ@?$CFd?6?$AA@" = linkonce_odr unnamed_addr constant [4 x i8] c"%d\0A\00", comdat, align 1

; This is just a minimal check to verify that main was handled by WinEHPrepare.
; CHECK: define i32 @main()
; CHECK: entry:
; CHECK:   call void (...) @llvm.frameescape(i32* [[X_PTR:\%.+]], i32* [[X2_PTR:\%.+]], i8* [[C2_PTR:\%.+]], i8* [[C3_PTR:\%.+]], i8* [[C_PTR:\%.+]])
; CHECK:   invoke void @_CxxThrowException
; CHECK: }

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %tmp = alloca i8, align 1
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %c = alloca i8, align 1
  %tmp3 = alloca i32, align 4
  %x = alloca i32, align 4
  %tmp20 = alloca i8, align 1
  %c28 = alloca i8, align 1
  %tmp34 = alloca i32, align 4
  %c48 = alloca i8, align 1
  %x56 = alloca i32, align 4
  store i32 0, i32* %retval
  store i8 97, i8* %tmp
  invoke void @_CxxThrowException(i8* %tmp, %eh.ThrowInfo* @_TI1D) #4
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.D.0
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.0
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  store i8* %1, i8** %exn.slot
  %2 = extractvalue { i8*, i32 } %0, 1
  store i32 %2, i32* %ehselector.slot
  br label %catch.dispatch

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, i32* %ehselector.slot
  %3 = call i32 @llvm.eh.typeid.for(i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.D.0 to i8*)) #2
  %matches = icmp eq i32 %sel, %3
  br i1 %matches, label %catch, label %catch.dispatch5

catch:                                            ; preds = %catch.dispatch
  %exn = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn, i8* %c) #2
  %4 = load i8, i8* %c, align 1
  %conv = sext i8 %4 to i32
  %call = invoke i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01??_C@_03PJCJOCBM@?$CFc?6?$AA@", i32 0, i32 0), i32 %conv)
          to label %invoke.cont unwind label %lpad2

invoke.cont:                                      ; preds = %catch
  call void @llvm.eh.endcatch() #2
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont
  store i32 1, i32* %tmp3
  %5 = bitcast i32* %tmp3 to i8*
  invoke void @_CxxThrowException(i8* %5, %eh.ThrowInfo* @_TI1H) #4
          to label %unreachable unwind label %lpad4

lpad2:                                            ; preds = %catch
  %6 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.0
          catch i8* null
  %7 = extractvalue { i8*, i32 } %6, 0
  store i8* %7, i8** %exn.slot
  %8 = extractvalue { i8*, i32 } %6, 1
  store i32 %8, i32* %ehselector.slot
  call void @llvm.eh.endcatch() #2
  br label %catch.dispatch5

lpad4:                                            ; preds = %try.cont
  %9 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.0
          catch i8* null
  %10 = extractvalue { i8*, i32 } %9, 0
  store i8* %10, i8** %exn.slot
  %11 = extractvalue { i8*, i32 } %9, 1
  store i32 %11, i32* %ehselector.slot
  br label %catch.dispatch5

catch.dispatch5:                                  ; preds = %lpad4, %lpad2, %catch.dispatch
  %sel6 = load i32, i32* %ehselector.slot
  %12 = call i32 @llvm.eh.typeid.for(i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*)) #2
  %matches7 = icmp eq i32 %sel6, %12
  br i1 %matches7, label %catch13, label %catch8

catch13:                                          ; preds = %catch.dispatch5
  %exn14 = load i8*, i8** %exn.slot
  %13 = bitcast i32* %x to i8*
  call void @llvm.eh.begincatch(i8* %exn14, i8* %13) #2
  %14 = load i32, i32* %x, align 4
  %call18 = invoke i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01??_C@_03PMGGPEJJ@?$CFd?6?$AA@", i32 0, i32 0), i32 %14)
          to label %invoke.cont17 unwind label %lpad16

invoke.cont17:                                    ; preds = %catch13
  call void @llvm.eh.endcatch() #2
  br label %try.cont19

try.cont19:                                       ; preds = %invoke.cont17, %invoke.cont11
  store i8 98, i8* %tmp20
  invoke void @_CxxThrowException(i8* %tmp20, %eh.ThrowInfo* @_TI1D) #4
          to label %unreachable unwind label %lpad21

catch8:                                           ; preds = %catch.dispatch5
  %exn9 = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn9, i8* null) #2
  %call12 = invoke i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @"\01??_C@_04MPPNMCOK@?4?4?4?6?$AA@", i32 0, i32 0))
          to label %invoke.cont11 unwind label %lpad10

invoke.cont11:                                    ; preds = %catch8
  call void @llvm.eh.endcatch() #2
  br label %try.cont19

lpad10:                                           ; preds = %catch8
  %15 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
  %16 = extractvalue { i8*, i32 } %15, 0
  store i8* %16, i8** %exn.slot
  %17 = extractvalue { i8*, i32 } %15, 1
  store i32 %17, i32* %ehselector.slot
  call void @llvm.eh.endcatch() #2
  br label %eh.resume

lpad16:                                           ; preds = %catch13
  %18 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
  %19 = extractvalue { i8*, i32 } %18, 0
  store i8* %19, i8** %exn.slot
  %20 = extractvalue { i8*, i32 } %18, 1
  store i32 %20, i32* %ehselector.slot
  call void @llvm.eh.endcatch() #2
  br label %eh.resume

lpad21:                                           ; preds = %try.cont19
  %21 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.D.0 to i8*)
          catch i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*)
          catch i8* null
  %22 = extractvalue { i8*, i32 } %21, 0
  store i8* %22, i8** %exn.slot
  %23 = extractvalue { i8*, i32 } %21, 1
  store i32 %23, i32* %ehselector.slot
  br label %catch.dispatch22

catch.dispatch22:                                 ; preds = %lpad21
  %sel23 = load i32, i32* %ehselector.slot
  %24 = call i32 @llvm.eh.typeid.for(i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.D.0 to i8*)) #2
  %matches24 = icmp eq i32 %sel23, %24
  br i1 %matches24, label %catch25, label %catch.dispatch36

catch25:                                          ; preds = %catch.dispatch22
  %exn26 = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn26, i8* %c28) #2
  %25 = load i8, i8* %c28, align 1
  %conv29 = sext i8 %25 to i32
  %call32 = invoke i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01??_C@_03PJCJOCBM@?$CFc?6?$AA@", i32 0, i32 0), i32 %conv29)
          to label %invoke.cont31 unwind label %lpad30

invoke.cont31:                                    ; preds = %catch25
  call void @llvm.eh.endcatch() #2
  br label %try.cont33

try.cont33:                                       ; preds = %invoke.cont31
  store i32 2, i32* %tmp34
  %26 = bitcast i32* %tmp34 to i8*
  invoke void @_CxxThrowException(i8* %26, %eh.ThrowInfo* @_TI1H) #4
          to label %unreachable unwind label %lpad35

lpad30:                                           ; preds = %catch25
  %27 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*)
          catch i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.D.0 to i8*)
          catch i8* null
  %28 = extractvalue { i8*, i32 } %27, 0
  store i8* %28, i8** %exn.slot
  %29 = extractvalue { i8*, i32 } %27, 1
  store i32 %29, i32* %ehselector.slot
  call void @llvm.eh.endcatch() #2
  br label %catch.dispatch36

lpad35:                                           ; preds = %try.cont33
  %30 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*)
          catch i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.D.0 to i8*)
          catch i8* null
  %31 = extractvalue { i8*, i32 } %30, 0
  store i8* %31, i8** %exn.slot
  %32 = extractvalue { i8*, i32 } %30, 1
  store i32 %32, i32* %ehselector.slot
  br label %catch.dispatch36

catch.dispatch36:                                 ; preds = %lpad35, %lpad30, %catch.dispatch22
  %sel37 = load i32, i32* %ehselector.slot
  %33 = call i32 @llvm.eh.typeid.for(i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*)) #2
  %matches38 = icmp eq i32 %sel37, %33
  br i1 %matches38, label %catch53, label %catch.fallthrough

catch53:                                          ; preds = %catch.dispatch36
  %exn54 = load i8*, i8** %exn.slot
  %34 = bitcast i32* %x56 to i8*
  call void @llvm.eh.begincatch(i8* %exn54, i8* %34) #2
  %35 = load i32, i32* %x56, align 4
  %call59 = invoke i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01??_C@_03PMGGPEJJ@?$CFd?6?$AA@", i32 0, i32 0), i32 %35)
          to label %invoke.cont58 unwind label %lpad57

invoke.cont58:                                    ; preds = %catch53
  call void @llvm.eh.endcatch() #2
  br label %try.cont60

try.cont60:                                       ; preds = %invoke.cont58, %invoke.cont51, %invoke.cont43
  ret i32 0

catch.fallthrough:                                ; preds = %catch.dispatch36
  %36 = call i32 @llvm.eh.typeid.for(i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.D.0 to i8*)) #2
  %matches39 = icmp eq i32 %sel37, %36
  br i1 %matches39, label %catch45, label %catch40

catch45:                                          ; preds = %catch.fallthrough
  %exn46 = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn46, i8* %c48) #2
  %37 = load i8, i8* %c48, align 1
  %conv49 = sext i8 %37 to i32
  %call52 = invoke i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01??_C@_03PJCJOCBM@?$CFc?6?$AA@", i32 0, i32 0), i32 %conv49)
          to label %invoke.cont51 unwind label %lpad50

invoke.cont51:                                    ; preds = %catch45
  call void @llvm.eh.endcatch() #2
  br label %try.cont60

catch40:                                          ; preds = %catch.fallthrough
  %exn41 = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn41, i8* null) #2
  %call44 = invoke i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @"\01??_C@_04MPPNMCOK@?4?4?4?6?$AA@", i32 0, i32 0))
          to label %invoke.cont43 unwind label %lpad42

invoke.cont43:                                    ; preds = %catch40
  call void @llvm.eh.endcatch() #2
  br label %try.cont60

lpad42:                                           ; preds = %catch40
  %38 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
  %39 = extractvalue { i8*, i32 } %38, 0
  store i8* %39, i8** %exn.slot
  %40 = extractvalue { i8*, i32 } %38, 1
  store i32 %40, i32* %ehselector.slot
  call void @llvm.eh.endcatch() #2
  br label %eh.resume

lpad50:                                           ; preds = %catch45
  %41 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
  %42 = extractvalue { i8*, i32 } %41, 0
  store i8* %42, i8** %exn.slot
  %43 = extractvalue { i8*, i32 } %41, 1
  store i32 %43, i32* %ehselector.slot
  call void @llvm.eh.endcatch() #2
  br label %eh.resume

lpad57:                                           ; preds = %catch53
  %44 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
  %45 = extractvalue { i8*, i32 } %44, 0
  store i8* %45, i8** %exn.slot
  %46 = extractvalue { i8*, i32 } %44, 1
  store i32 %46, i32* %ehselector.slot
  call void @llvm.eh.endcatch() #2
  br label %eh.resume

eh.resume:                                        ; preds = %lpad57, %lpad50, %lpad42, %lpad16, %lpad10
  %exn61 = load i8*, i8** %exn.slot
  %sel62 = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn61, 0
  %lpad.val63 = insertvalue { i8*, i32 } %lpad.val, i32 %sel62, 1
  resume { i8*, i32 } %lpad.val63

unreachable:                                      ; preds = %try.cont33, %try.cont19, %try.cont, %entry
  unreachable
}

declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #2

declare i32 @printf(i8*, ...) #3

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (trunk 235214) (llvm/trunk 235213)"}
