; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; C++ EH rethrows are interesting, because they are calls to noreturn
; functions. There *must* be some code after the call instruction that doesn't
; look like an epilogue. We use int3 to be consistent with MSVC.

; Based on this C++ source:
; int main() {
;   try {
;     throw 42;
;   } catch (int) {
;     try {
;       throw;
;     } catch (int) {
;     }
;   }
;   return 0;
; }

; ModuleID = 't.cpp'
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

define i32 @main() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %tmp = alloca i32, align 4
  store i32 42, i32* %tmp, align 4
  %0 = bitcast i32* %tmp to i8*
  invoke void @_CxxThrowException(i8* %0, %eh.ThrowInfo* nonnull @_TI1H) #1
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %1 = catchpad [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
          to label %catch unwind label %catchendblock

catch:                                            ; preds = %catch.dispatch
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) #1
          to label %unreachable unwind label %catch.dispatch.1

catch.dispatch.1:                                 ; preds = %catch
  %2 = catchpad [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
          to label %catch.3 unwind label %catchendblock.2

catch.3:                                          ; preds = %catch.dispatch.1
  catchret %2 to label %try.cont

try.cont:                                         ; preds = %catch.3
  catchret %1 to label %try.cont.5

try.cont.5:                                       ; preds = %try.cont
  ret i32 0

catchendblock.2:                                  ; preds = %catch.dispatch.1
  catchendpad unwind label %catchendblock

catchendblock:                                    ; preds = %catchendblock.2, %catch.dispatch
  catchendpad unwind to caller

unreachable:                                      ; preds = %catch, %entry
  unreachable
}

declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)

declare i32 @__CxxFrameHandler3(...)

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn }

; CHECK: main:
; CHECK: .seh_proc main
; CHECK: movl $42,
; CHECK-DAG: leaq {{.*}}, %rcx
; CHECK-DAG: leaq _TI1H(%rip), %rdx
; CHECK: callq _CxxThrowException
; CHECK-NEXT: int3

; CHECK: "?catch$1@?0?main@4HA":
; CHECK: .seh_proc "?catch$1@?0?main@4HA"
; CHECK-DAG: xorl %ecx, %ecx
; CHECK-DAG: xorl %edx, %edx
; CHECK: callq _CxxThrowException
; CHECK-NEXT: int3
