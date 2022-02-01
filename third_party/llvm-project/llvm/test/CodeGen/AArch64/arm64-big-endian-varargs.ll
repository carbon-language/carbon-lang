; RUN: llc < %s | FileCheck %s

; Vararg saving must save Q registers using the equivalent of STR/STP.

target datalayout = "E-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64_be-arm-none-eabi"

%struct.__va_list = type { i8*, i8*, i8*, i32, i32 }

declare void @llvm.va_start(i8*) nounwind
declare void @llvm.va_end(i8*) nounwind

define double @callee(i32 %a, ...) {
; CHECK: stp
; CHECK: stp
; CHECK: stp
; CHECK: stp
; CHECK: stp
; CHECK: stp
entry:
  %vl = alloca %struct.__va_list, align 8
  %vl1 = bitcast %struct.__va_list* %vl to i8*
  call void @llvm.va_start(i8* %vl1)
  %vr_offs_p = getelementptr inbounds %struct.__va_list, %struct.__va_list* %vl, i64 0, i32 4
  %vr_offs = load i32, i32* %vr_offs_p, align 4
  %0 = icmp sgt i32 %vr_offs, -1
  br i1 %0, label %vaarg.on_stack, label %vaarg.maybe_reg

vaarg.maybe_reg:                                  ; preds = %entry
  %new_reg_offs = add i32 %vr_offs, 16
  store i32 %new_reg_offs, i32* %vr_offs_p, align 4
  %inreg = icmp slt i32 %new_reg_offs, 1
  br i1 %inreg, label %vaarg.in_reg, label %vaarg.on_stack

vaarg.in_reg:                                     ; preds = %vaarg.maybe_reg
  %reg_top_p = getelementptr inbounds %struct.__va_list, %struct.__va_list* %vl, i64 0, i32 2
  %reg_top = load i8*, i8** %reg_top_p, align 8
  %1 = sext i32 %vr_offs to i64
  %2 = getelementptr i8, i8* %reg_top, i64 %1
  %3 = ptrtoint i8* %2 to i64
  %align_be = add i64 %3, 8
  %4 = inttoptr i64 %align_be to i8*
  br label %vaarg.end

vaarg.on_stack:                                   ; preds = %vaarg.maybe_reg, %entry
  %stack_p = getelementptr inbounds %struct.__va_list, %struct.__va_list* %vl, i64 0, i32 0
  %stack = load i8*, i8** %stack_p, align 8
  %new_stack = getelementptr i8, i8* %stack, i64 8
  store i8* %new_stack, i8** %stack_p, align 8
  br label %vaarg.end

vaarg.end:                                        ; preds = %vaarg.on_stack, %vaarg.in_reg
  %.sink = phi i8* [ %4, %vaarg.in_reg ], [ %stack, %vaarg.on_stack ]
  %5 = bitcast i8* %.sink to double*
  %6 = load double, double* %5, align 8
  call void @llvm.va_end(i8* %vl1)
  ret double %6
}
