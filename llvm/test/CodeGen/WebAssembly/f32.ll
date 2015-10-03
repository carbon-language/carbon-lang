; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 32-bit floating-point operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare float @llvm.fabs.f32(float)
declare float @llvm.copysign.f32(float, float)
declare float @llvm.sqrt.f32(float)
declare float @llvm.ceil.f32(float)
declare float @llvm.floor.f32(float)
declare float @llvm.trunc.f32(float)
declare float @llvm.nearbyint.f32(float)
declare float @llvm.rint.f32(float)

; CHECK-LABEL: (func $fadd32
; CHECK-NEXT: (param f32) (param f32) (result f32)
; CHECK-NEXT: (set_local @0 (argument 1))
; CHECK-NEXT: (set_local @1 (argument 0))
; CHECK-NEXT: (set_local @2 (fadd @1 @0))
; CHECK-NEXT: (return @2)
define float @fadd32(float %x, float %y) {
  %a = fadd float %x, %y
  ret float %a
}

; CHECK-LABEL: (func $fsub32
; CHECK: (set_local @2 (fsub @1 @0))
define float @fsub32(float %x, float %y) {
  %a = fsub float %x, %y
  ret float %a
}

; CHECK-LABEL: (func $fmul32
; CHECK: (set_local @2 (fmul @1 @0))
define float @fmul32(float %x, float %y) {
  %a = fmul float %x, %y
  ret float %a
}

; CHECK-LABEL: (func $fdiv32
; CHECK: (set_local @2 (fdiv @1 @0))
define float @fdiv32(float %x, float %y) {
  %a = fdiv float %x, %y
  ret float %a
}

; CHECK-LABEL: (func $fabs32
; CHECK: (set_local @1 (fabs @0))
define float @fabs32(float %x) {
  %a = call float @llvm.fabs.f32(float %x)
  ret float %a
}

; CHECK-LABEL: (func $fneg32
; CHECK: (set_local @1 (fneg @0))
define float @fneg32(float %x) {
  %a = fsub float -0., %x
  ret float %a
}

; CHECK-LABEL: (func $copysign32
; CHECK: (set_local @2 (copysign @1 @0))
define float @copysign32(float %x, float %y) {
  %a = call float @llvm.copysign.f32(float %x, float %y)
  ret float %a
}

; CHECK-LABEL: (func $sqrt32
; CHECK: (set_local @1 (sqrt @0))
define float @sqrt32(float %x) {
  %a = call float @llvm.sqrt.f32(float %x)
  ret float %a
}

; CHECK-LABEL: (func $ceil32
; CHECK: (set_local @1 (ceil @0))
define float @ceil32(float %x) {
  %a = call float @llvm.ceil.f32(float %x)
  ret float %a
}

; CHECK-LABEL: (func $floor32
; CHECK: (set_local @1 (floor @0))
define float @floor32(float %x) {
  %a = call float @llvm.floor.f32(float %x)
  ret float %a
}

; CHECK-LABEL: (func $trunc32
; CHECK: (set_local @1 (trunc @0))
define float @trunc32(float %x) {
  %a = call float @llvm.trunc.f32(float %x)
  ret float %a
}

; CHECK-LABEL: (func $nearest32
; CHECK: (set_local @1 (nearest @0))
define float @nearest32(float %x) {
  %a = call float @llvm.nearbyint.f32(float %x)
  ret float %a
}

; CHECK-LABEL: (func $nearest32_via_rint
; CHECK: (set_local @1 (nearest @0))
define float @nearest32_via_rint(float %x) {
  %a = call float @llvm.rint.f32(float %x)
  ret float %a
}
