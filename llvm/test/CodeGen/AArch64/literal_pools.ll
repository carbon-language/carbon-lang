; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -code-model=large | FileCheck --check-prefix=CHECK-LARGE %s

@var32 = global i32 0
@var64 = global i64 0

define void @foo() {
; CHECK: foo:
    %val32 = load i32* @var32
    %val64 = load i64* @var64

    %val32_lit32 = and i32 %val32, 123456785
    store volatile i32 %val32_lit32, i32* @var32
; CHECK: adrp x[[LITBASE:[0-9]+]], [[CURLIT:.LCPI0_[0-9]+]]
; CHECK: ldr {{w[0-9]+}}, [x[[LITBASE]], #:lo12:[[CURLIT]]]

; CHECK-LARGE: movz x[[LITADDR:[0-9]+]], #:abs_g3:[[CURLIT:.LCPI0_[0-9]+]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g2_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g1_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g0_nc:[[CURLIT]]
; CHECK-LARGE: ldr {{w[0-9]+}}, [x[[LITADDR]]]

    %val64_lit32 = and i64 %val64, 305402420
    store volatile i64 %val64_lit32, i64* @var64
; CHECK: adrp x[[LITBASE:[0-9]+]], [[CURLIT:.LCPI0_[0-9]+]]
; CHECK: ldr {{w[0-9]+}}, [x[[LITBASE]], #:lo12:[[CURLIT]]]

; CHECK-LARGE: movz x[[LITADDR:[0-9]+]], #:abs_g3:[[CURLIT:.LCPI0_[0-9]+]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g2_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g1_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g0_nc:[[CURLIT]]
; CHECK-LARGE: ldr {{w[0-9]+}}, [x[[LITADDR]]]

    %val64_lit32signed = and i64 %val64, -12345678
    store volatile i64 %val64_lit32signed, i64* @var64
; CHECK: adrp x[[LITBASE:[0-9]+]], [[CURLIT:.LCPI0_[0-9]+]]
; CHECK: ldrsw {{x[0-9]+}}, [x[[LITBASE]], #:lo12:[[CURLIT]]]

; CHECK-LARGE: movz x[[LITADDR:[0-9]+]], #:abs_g3:[[CURLIT:.LCPI0_[0-9]+]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g2_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g1_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g0_nc:[[CURLIT]]
; CHECK-LARGE: ldrsw {{x[0-9]+}}, [x[[LITADDR]]]

    %val64_lit64 = and i64 %val64, 1234567898765432
    store volatile i64 %val64_lit64, i64* @var64
; CHECK: adrp x[[LITBASE:[0-9]+]], [[CURLIT:.LCPI0_[0-9]+]]
; CHECK: ldr {{x[0-9]+}}, [x[[LITBASE]], #:lo12:[[CURLIT]]]

; CHECK-LARGE: movz x[[LITADDR:[0-9]+]], #:abs_g3:[[CURLIT:.LCPI0_[0-9]+]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g2_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g1_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g0_nc:[[CURLIT]]
; CHECK-LARGE: ldr {{x[0-9]+}}, [x[[LITADDR]]]

    ret void
}

@varfloat = global float 0.0
@vardouble = global double 0.0

define void @floating_lits() {
; CHECK: floating_lits:

  %floatval = load float* @varfloat
  %newfloat = fadd float %floatval, 128.0
; CHECK: adrp x[[LITBASE:[0-9]+]], [[CURLIT:.LCPI1_[0-9]+]]
; CHECK: ldr {{s[0-9]+}}, [x[[LITBASE]], #:lo12:[[CURLIT]]]
; CHECK: fadd

; CHECK-LARGE: movz x[[LITADDR:[0-9]+]], #:abs_g3:[[CURLIT:.LCPI1_[0-9]+]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g2_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g1_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g0_nc:[[CURLIT]]
; CHECK-LARGE: ldr {{s[0-9]+}}, [x[[LITADDR]]]
; CHECK-LARGE: fadd

  store float %newfloat, float* @varfloat

  %doubleval = load double* @vardouble
  %newdouble = fadd double %doubleval, 129.0
; CHECK: adrp x[[LITBASE:[0-9]+]], [[CURLIT:.LCPI1_[0-9]+]]
; CHECK: ldr {{d[0-9]+}}, [x[[LITBASE]], #:lo12:[[CURLIT]]]
; CHECK: fadd

; CHECK-LARGE: movz x[[LITADDR:[0-9]+]], #:abs_g3:[[CURLIT:.LCPI1_[0-9]+]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g2_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g1_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g0_nc:[[CURLIT]]
; CHECK-LARGE: ldr {{d[0-9]+}}, [x[[LITADDR]]]

  store double %newdouble, double* @vardouble

  ret void
}
