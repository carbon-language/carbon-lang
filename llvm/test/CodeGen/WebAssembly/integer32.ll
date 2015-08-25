; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 32-bit integer operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i32 @llvm.ctpop.i32(i32)

; CHECK-LABEL: (func $add32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (add @1 @0))
; CHECK-NEXT: (return @2)
define i32 @add32(i32 %x, i32 %y) {
  %a = add i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $sub32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (sub @1 @0))
; CHECK-NEXT: (return @2)
define i32 @sub32(i32 %x, i32 %y) {
  %a = sub i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $mul32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (mul @1 @0))
; CHECK-NEXT: (return @2)
define i32 @mul32(i32 %x, i32 %y) {
  %a = mul i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $sdiv32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (sdiv @1 @0))
; CHECK-NEXT: (return @2)
define i32 @sdiv32(i32 %x, i32 %y) {
  %a = sdiv i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $udiv32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (udiv @1 @0))
; CHECK-NEXT: (return @2)
define i32 @udiv32(i32 %x, i32 %y) {
  %a = udiv i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $srem32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (srem @1 @0))
; CHECK-NEXT: (return @2)
define i32 @srem32(i32 %x, i32 %y) {
  %a = srem i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $urem32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (urem @1 @0))
; CHECK-NEXT: (return @2)
define i32 @urem32(i32 %x, i32 %y) {
  %a = urem i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $and32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (and @1 @0))
; CHECK-NEXT: (return @2)
define i32 @and32(i32 %x, i32 %y) {
  %a = and i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $ior32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (ior @1 @0))
; CHECK-NEXT: (return @2)
define i32 @ior32(i32 %x, i32 %y) {
  %a = or i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $xor32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (xor @1 @0))
; CHECK-NEXT: (return @2)
define i32 @xor32(i32 %x, i32 %y) {
  %a = xor i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $shl32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (shl @1 @0))
; CHECK-NEXT: (return @2)
define i32 @shl32(i32 %x, i32 %y) {
  %a = shl i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $shr32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (shr @1 @0))
; CHECK-NEXT: (return @2)
define i32 @shr32(i32 %x, i32 %y) {
  %a = lshr i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $sar32
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (sar @1 @0))
; CHECK-NEXT: (return @2)
define i32 @sar32(i32 %x, i32 %y) {
  %a = ashr i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: (func $clz32
; CHECK-NEXT: (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (clz @0))
; CHECK-NEXT: (return @1)
define i32 @clz32(i32 %x) {
  %a = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %a
}

; CHECK-LABEL: (func $clz32_zero_undef
; CHECK-NEXT: (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (clz @0))
; CHECK-NEXT: (return @1)
define i32 @clz32_zero_undef(i32 %x) {
  %a = call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  ret i32 %a
}

; CHECK-LABEL: (func $ctz32
; CHECK-NEXT: (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (ctz @0))
; CHECK-NEXT: (return @1)
define i32 @ctz32(i32 %x) {
  %a = call i32 @llvm.cttz.i32(i32 %x, i1 false)
  ret i32 %a
}

; CHECK-LABEL: (func $ctz32_zero_undef
; CHECK-NEXT: (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (ctz @0))
; CHECK-NEXT: (return @1)
define i32 @ctz32_zero_undef(i32 %x) {
  %a = call i32 @llvm.cttz.i32(i32 %x, i1 true)
  ret i32 %a
}

; CHECK-LABEL: (func $popcnt32
; CHECK-NEXT: (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (popcnt @0))
; CHECK-NEXT: (return @1)
define i32 @popcnt32(i32 %x) {
  %a = call i32 @llvm.ctpop.i32(i32 %x)
  ret i32 %a
}
