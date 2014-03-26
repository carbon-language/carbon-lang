; RUN: llc -mcpu=pwr7 -mattr=+vsx < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define double @test1(double %a, double %b) {
entry:
  %v = fmul double %a, %b
  ret double %v

; CHECK-LABEL: @test1
; CHECK: xsmuldp 1, 1, 2
; CHECK: blr
}

define double @test2(double %a, double %b) {
entry:
  %v = fdiv double %a, %b
  ret double %v

; CHECK-LABEL: @test2
; CHECK: xsdivdp 1, 1, 2
; CHECK: blr
}

define double @test3(double %a, double %b) {
entry:
  %v = fadd double %a, %b
  ret double %v

; CHECK-LABEL: @test3
; CHECK: xsadddp 1, 1, 2
; CHECK: blr
}

define <2 x double> @test4(<2 x double> %a, <2 x double> %b) {
entry:
  %v = fadd <2 x double> %a, %b
  ret <2 x double> %v

; CHECK-LABEL: @test4
; CHECK: xvadddp 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test5(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = xor <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-LABEL: @test5
; CHECK: xxlxor 34, 34, 35
; CHECK: blr
}

define <8 x i16> @test6(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = xor <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-LABEL: @test6
; CHECK: xxlxor 34, 34, 35
; CHECK: blr
}

define <16 x i8> @test7(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = xor <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-LABEL: @test7
; CHECK: xxlxor 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test8(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = or <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-LABEL: @test8
; CHECK: xxlor 34, 34, 35
; CHECK: blr
}

define <8 x i16> @test9(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = or <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-LABEL: @test9
; CHECK: xxlor 34, 34, 35
; CHECK: blr
}

define <16 x i8> @test10(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = or <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-LABEL: @test10
; CHECK: xxlor 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test11(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = and <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-LABEL: @test11
; CHECK: xxland 34, 34, 35
; CHECK: blr
}

define <8 x i16> @test12(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = and <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-LABEL: @test12
; CHECK: xxland 34, 34, 35
; CHECK: blr
}

define <16 x i8> @test13(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = and <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-LABEL: @test13
; CHECK: xxland 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test14(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = or <4 x i32> %a, %b
  %w = xor <4 x i32> %v, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %w

; CHECK-LABEL: @test14
; CHECK: xxlnor 34, 34, 35
; CHECK: blr
}

define <8 x i16> @test15(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = or <8 x i16> %a, %b
  %w = xor <8 x i16> %v, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  ret <8 x i16> %w

; CHECK-LABEL: @test15
; CHECK: xxlnor 34, 34, 35
; CHECK: blr
}

define <16 x i8> @test16(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = or <16 x i8> %a, %b
  %w = xor <16 x i8> %v, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  ret <16 x i8> %w

; CHECK-LABEL: @test16
; CHECK: xxlnor 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test17(<4 x i32> %a, <4 x i32> %b) {
entry:
  %w = xor <4 x i32> %b, <i32 -1, i32 -1, i32 -1, i32 -1>
  %v = and <4 x i32> %a, %w
  ret <4 x i32> %v

; CHECK-LABEL: @test17
; CHECK: xxlandc 34, 34, 35
; CHECK: blr
}

define <8 x i16> @test18(<8 x i16> %a, <8 x i16> %b) {
entry:
  %w = xor <8 x i16> %b, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %v = and <8 x i16> %a, %w
  ret <8 x i16> %v

; CHECK-LABEL: @test18
; CHECK: xxlandc 34, 34, 35
; CHECK: blr
}

define <16 x i8> @test19(<16 x i8> %a, <16 x i8> %b) {
entry:
  %w = xor <16 x i8> %b, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %v = and <16 x i8> %a, %w
  ret <16 x i8> %v

; CHECK-LABEL: @test19
; CHECK: xxlandc 34, 34, 35
; CHECK: blr
}

