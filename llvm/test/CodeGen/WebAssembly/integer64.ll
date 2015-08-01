; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit integer operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i64 @llvm.ctlz.i64(i64, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i64 @llvm.ctpop.i64(i64)

; CHECK-LABEL: add64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (ADD_I64 @1 @0))
; CHECK-NEXT: (return @2)
define i64 @add64(i64 %x, i64 %y) {
  %a = add i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: sub64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (SUB_I64 @1 @0))
; CHECK-NEXT: (return @2)
define i64 @sub64(i64 %x, i64 %y) {
  %a = sub i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: mul64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (MUL_I64 @1 @0))
; CHECK-NEXT: (return @2)
define i64 @mul64(i64 %x, i64 %y) {
  %a = mul i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: sdiv64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (SDIV_I64 @1 @0))
; CHECK-NEXT: (return @2)
define i64 @sdiv64(i64 %x, i64 %y) {
  %a = sdiv i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: udiv64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (UDIV_I64 @1 @0))
; CHECK-NEXT: (return @2)
define i64 @udiv64(i64 %x, i64 %y) {
  %a = udiv i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: srem64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (SREM_I64 @1 @0))
; CHECK-NEXT: (return @2)
define i64 @srem64(i64 %x, i64 %y) {
  %a = srem i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: urem64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (UREM_I64 @1 @0))
; CHECK-NEXT: (return @2)
define i64 @urem64(i64 %x, i64 %y) {
  %a = urem i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: and64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (AND_I64 @1 @0))
; CHECK-NEXT: (return @2)
define i64 @and64(i64 %x, i64 %y) {
  %a = and i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: ior64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (IOR_I64 @1 @0))
; CHECK-NEXT: (return @2)
define i64 @ior64(i64 %x, i64 %y) {
  %a = or i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: xor64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (XOR_I64 @1 @0))
; CHECK-NEXT: (return @2)
define i64 @xor64(i64 %x, i64 %y) {
  %a = xor i64 %x, %y
  ret i64 %a
}

; FIXME: 64-bit shifts have an extra truncate of the input shift value, which
;        WebAssembly hasn't taught isel to match yet. Fix with
;        getScalarShiftAmountTy.

; C;HECK-LABEL: shl64:
; C;HECK-NEXT: (setlocal @0 (argument 1))
; C;HECK-NEXT: (setlocal @1 (argument 0))
; C;HECK-NEXT: (setlocal @2 (SHL_I64 @1 @0))
; C;HECK-NEXT: (return @2)
;define i64 @shl64(i64 %x, i64 %y) {
;  %a = shl i64 %x, %y
;  ret i64 %a
;}

; C;HECK-LABEL: shr64:
; C;HECK-NEXT: (setlocal @0 (argument 1))
; C;HECK-NEXT: (setlocal @1 (argument 0))
; C;HECK-NEXT: (setlocal @2 (SHR_I64 @1 @0))
; C;HECK-NEXT: (return @2)
;define i64 @shr64(i64 %x, i64 %y) {
;  %a = lshr i64 %x, %y
;  ret i64 %a
;}

; C;HECK-LABEL: sar64:
; C;HECK-NEXT: (setlocal @0 (argument 1))
; C;HECK-NEXT: (setlocal @1 (argument 0))
; C;HECK-NEXT: (setlocal @2 (SAR_I64 @1 @0))
; C;HECK-NEXT: (return @2)
;define i64 @sar64(i64 %x, i64 %y) {
;  %a = ashr i64 %x, %y
;  ret i64 %a
;}

; CHECK-LABEL: clz64:
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (CLZ_I64 @0))
; CHECK-NEXT: (return @1)
define i64 @clz64(i64 %x) {
  %a = call i64 @llvm.ctlz.i64(i64 %x, i1 false)
  ret i64 %a
}

; CHECK-LABEL: ctz64:
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (CTZ_I64 @0))
; CHECK-NEXT: (return @1)
define i64 @ctz64(i64 %x) {
  %a = call i64 @llvm.cttz.i64(i64 %x, i1 false)
  ret i64 %a
}

; CHECK-LABEL: popcnt64:
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (POPCNT_I64 @0))
; CHECK-NEXT: (return @1)
define i64 @popcnt64(i64 %x) {
  %a = call i64 @llvm.ctpop.i64(i64 %x)
  ret i64 %a
}
