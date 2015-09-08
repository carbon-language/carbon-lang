; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit integer operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i64 @llvm.ctlz.i64(i64, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i64 @llvm.ctpop.i64(i64)

; CHECK-LABEL: (func $add64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (add @1 @0))
; CHECK-NEXT: (return @2)
define i64 @add64(i64 %x, i64 %y) {
  %a = add i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $sub64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (sub @1 @0))
; CHECK-NEXT: (return @2)
define i64 @sub64(i64 %x, i64 %y) {
  %a = sub i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $mul64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (mul @1 @0))
; CHECK-NEXT: (return @2)
define i64 @mul64(i64 %x, i64 %y) {
  %a = mul i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $sdiv64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (sdiv @1 @0))
; CHECK-NEXT: (return @2)
define i64 @sdiv64(i64 %x, i64 %y) {
  %a = sdiv i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $udiv64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (udiv @1 @0))
; CHECK-NEXT: (return @2)
define i64 @udiv64(i64 %x, i64 %y) {
  %a = udiv i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $srem64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (srem @1 @0))
; CHECK-NEXT: (return @2)
define i64 @srem64(i64 %x, i64 %y) {
  %a = srem i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $urem64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (urem @1 @0))
; CHECK-NEXT: (return @2)
define i64 @urem64(i64 %x, i64 %y) {
  %a = urem i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $and64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (and @1 @0))
; CHECK-NEXT: (return @2)
define i64 @and64(i64 %x, i64 %y) {
  %a = and i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $ior64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (ior @1 @0))
; CHECK-NEXT: (return @2)
define i64 @ior64(i64 %x, i64 %y) {
  %a = or i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $xor64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (xor @1 @0))
; CHECK-NEXT: (return @2)
define i64 @xor64(i64 %x, i64 %y) {
  %a = xor i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $shl64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (shl @1 @0))
; CHECK-NEXT: (return @2)
define i64 @shl64(i64 %x, i64 %y) {
  %a = shl i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $shr64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (shr_u @1 @0))
; CHECK-NEXT: (return @2)
define i64 @shr64(i64 %x, i64 %y) {
  %a = lshr i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $sar64
; CHECK-NEXT: (param i64) (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (shr_s @1 @0))
; CHECK-NEXT: (return @2)
define i64 @sar64(i64 %x, i64 %y) {
  %a = ashr i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: (func $clz64
; CHECK-NEXT: (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (clz @0))
; CHECK-NEXT: (return @1)
define i64 @clz64(i64 %x) {
  %a = call i64 @llvm.ctlz.i64(i64 %x, i1 false)
  ret i64 %a
}

; CHECK-LABEL: (func $clz64_zero_undef
; CHECK-NEXT: (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (clz @0))
; CHECK-NEXT: (return @1)
define i64 @clz64_zero_undef(i64 %x) {
  %a = call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  ret i64 %a
}

; CHECK-LABEL: (func $ctz64
; CHECK-NEXT: (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (ctz @0))
; CHECK-NEXT: (return @1)
define i64 @ctz64(i64 %x) {
  %a = call i64 @llvm.cttz.i64(i64 %x, i1 false)
  ret i64 %a
}

; CHECK-LABEL: (func $ctz64_zero_undef
; CHECK-NEXT: (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (ctz @0))
; CHECK-NEXT: (return @1)
define i64 @ctz64_zero_undef(i64 %x) {
  %a = call i64 @llvm.cttz.i64(i64 %x, i1 true)
  ret i64 %a
}

; CHECK-LABEL: (func $popcnt64
; CHECK-NEXT: (param i64) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (popcnt @0))
; CHECK-NEXT: (return @1)
define i64 @popcnt64(i64 %x) {
  %a = call i64 @llvm.ctpop.i64(i64 %x)
  ret i64 %a
}
