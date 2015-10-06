; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit floating-point comparison operations assemble as
; expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: ord_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: @1{{$}}
; CHECK-NEXT: set_local @3, pop{{$}}
; CHECK-NEXT: eq @3, @3{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
; CHECK-NEXT: eq @2, @2{{$}}
; CHECK-NEXT: set_local @5, pop{{$}}
; CHECK-NEXT: and @5, @4{{$}}
; CHECK-NEXT: set_local @6, pop{{$}}
; CHECK-NEXT: return @6{{$}}
define i32 @ord_f64(double %x, double %y) {
  %a = fcmp ord double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uno_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: @1{{$}}
; CHECK-NEXT: set_local @3, pop{{$}}
; CHECK-NEXT: ne @3, @3{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
; CHECK-NEXT: ne @2, @2{{$}}
; CHECK-NEXT: set_local @5, pop{{$}}
; CHECK-NEXT: ior @5, @4{{$}}
; CHECK-NEXT: set_local @6, pop{{$}}
; CHECK-NEXT: return @6{{$}}
define i32 @uno_f64(double %x, double %y) {
  %a = fcmp uno double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oeq_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: @1{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @3, pop{{$}}
; CHECK-NEXT: eq @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
; CHECK-NEXT: return @4{{$}}
define i32 @oeq_f64(double %x, double %y) {
  %a = fcmp oeq double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: une_f64:
; CHECK: ne @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @une_f64(double %x, double %y) {
  %a = fcmp une double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: olt_f64:
; CHECK: lt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @olt_f64(double %x, double %y) {
  %a = fcmp olt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ole_f64:
; CHECK: le @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ole_f64(double %x, double %y) {
  %a = fcmp ole double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ogt_f64:
; CHECK: gt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ogt_f64(double %x, double %y) {
  %a = fcmp ogt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oge_f64:
; CHECK: ge @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @oge_f64(double %x, double %y) {
  %a = fcmp oge double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; Expanded comparisons, which also check for NaN.

; CHECK-LABEL: ueq_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: @1{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @3, pop{{$}}
; CHECK-NEXT: eq @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
; CHECK-NEXT: ne @2, @2{{$}}
; CHECK-NEXT: set_local @5, pop{{$}}
; CHECK-NEXT: ne @3, @3{{$}}
; CHECK-NEXT: set_local @6, pop{{$}}
; CHECK-NEXT: ior @6, @5{{$}}
; CHECK-NEXT: set_local @7, pop{{$}}
; CHECK-NEXT: ior @4, @7{{$}}
; CHECK-NEXT: set_local @8, pop{{$}}
; CHECK-NEXT: return @8{{$}}
define i32 @ueq_f64(double %x, double %y) {
  %a = fcmp ueq double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: one_f64:
; CHECK: ne @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @one_f64(double %x, double %y) {
  %a = fcmp one double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_f64:
; CHECK: lt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ult_f64(double %x, double %y) {
  %a = fcmp ult double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_f64:
; CHECK: le @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ule_f64(double %x, double %y) {
  %a = fcmp ule double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_f64:
; CHECK: gt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ugt_f64(double %x, double %y) {
  %a = fcmp ugt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_f64:
; CHECK: ge @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @uge_f64(double %x, double %y) {
  %a = fcmp uge double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
