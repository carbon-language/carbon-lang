; RUN: llc < %s | FileCheck %s

; Verify that we get the right frame escape label when the catch comes after the
; parent function.

; This test case is equivalent to:
; int main() {
;   try {
;     throw 42;
;   } catch (int e) {
;     printf("e: %d\n", e);
;   }
; }

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }
%eh.CatchHandlerType = type { i32, i8* }

$"\01??_R0H@8" = comdat any

$"_CT??_R0H@84" = comdat any

$_CTA1H = comdat any

$_TI1H = comdat any

$"\01??_C@_06PNOAJMHG@e?3?5?$CFd?6?$AA@" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@__ImageBase = external constant i8
@"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
@_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableType* @"_CT??_R0H@84" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableTypeArray.1* @_CTA1H to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@llvm.eh.handlertype.H.0 = private unnamed_addr constant %eh.CatchHandlerType { i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*) }, section "llvm.metadata"
@"\01??_C@_06PNOAJMHG@e?3?5?$CFd?6?$AA@" = linkonce_odr unnamed_addr constant [7 x i8] c"e: %d\0A\00", comdat, align 1

declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)

; Function Attrs: uwtable
define i32 @main() #1 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %tmp.i = alloca i32, align 4
  %e = alloca i32, align 4
  %0 = bitcast i32* %tmp.i to i8*
  store i32 42, i32* %tmp.i, align 4, !tbaa !2
  call void (...) @llvm.localescape(i32* %e)
  invoke void @_CxxThrowException(i8* %0, %eh.ThrowInfo* @_TI1H) #6
          to label %.noexc unwind label %lpad1

.noexc:                                           ; preds = %entry
  unreachable

lpad1:                                            ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.0
  %recover = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*), i32 0, i8* (i8*, i8*)* @main.catch)
  indirectbr i8* %recover, [label %try.cont.split]

try.cont.split:                                   ; preds = %lpad1
  ret i32 0
}

; CHECK-LABEL: main:
; CHECK:        .seh_handlerdata
; CHECK:        .long   ($cppxdata$main)@IMGREL

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #3

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #4

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #3

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #3

; Function Attrs: nounwind
declare i8* @llvm.eh.actions(...) #3

define internal i8* @main.catch(i8*, i8*) #5 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %e.i8 = call i8* @llvm.localrecover(i8* bitcast (i32 ()* @main to i8*), i8* %1, i32 0)
  %e = bitcast i8* %e.i8 to i32*
  %2 = bitcast i32* %e to i8*
  %3 = load i32, i32* %e, align 4, !tbaa !2
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @"\01??_C@_06PNOAJMHG@e?3?5?$CFd?6?$AA@", i64 0, i64 0), i32 %3)
  invoke void @llvm.donothing()
          to label %entry.split unwind label %stub

entry.split:                                      ; preds = %entry
  ret i8* blockaddress(@main, %try.cont.split)

stub:                                             ; preds = %entry
  %4 = landingpad { i8*, i32 }
          cleanup
  %recover = call i8* (...) @llvm.eh.actions()
  unreachable
}

; CHECK-LABEL: main.catch:
; CHECK:        .seh_handlerdata
; CHECK:        .long   ($cppxdata$main)@IMGREL

; CHECK: .align 4
; CHECK-NEXT: $cppxdata$main:
; CHECK-NEXT:         .long   429065506
; CHECK-NEXT:         .long   2
; CHECK-NEXT:         .long   ($stateUnwindMap$main)@IMGREL
; CHECK-NEXT:         .long   1
; CHECK-NEXT:         .long   ($tryMap$main)@IMGREL
; CHECK-NEXT:         .long   3
; CHECK-NEXT:         .long   ($ip2state$main)@IMGREL
; CHECK-NEXT:         .long   40
; CHECK-NEXT:         .long   0
; CHECK-NEXT:         .long   1

; Make sure we get the right frame escape label.

; CHECK: $handlerMap$0$main:
; CHECK-NEXT:         .long   0
; CHECK-NEXT:         .long   "??_R0H@8"@IMGREL
; CHECK-NEXT:         .long   .Lmain$frame_escape_0
; CHECK-NEXT:         .long   main.catch@IMGREL
; CHECK-NEXT:         .long   .Lmain.catch$parent_frame_offset

; Function Attrs: nounwind readnone
declare void @llvm.donothing() #2

; Function Attrs: nounwind
declare void @llvm.localescape(...) #3

; Function Attrs: nounwind readnone
declare i8* @llvm.localrecover(i8*, i8*, i32) #2

attributes #0 = { noreturn uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" "wineh-parent"="main" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }
attributes #4 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { "wineh-parent"="main" }
attributes #6 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

