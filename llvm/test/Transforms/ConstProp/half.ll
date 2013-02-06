; RUN: opt -constprop -S < %s | FileCheck %s

; CHECK: fabs_call
define half @fabs_call() {
; CHECK: ret half 0xH5140
  %x = call half @llvm.fabs.f16(half -42.0)
  ret half %x
}
declare half @llvm.fabs.f16(half %x)

; CHECK: exp_call
define half @exp_call() {
; CHECK: ret half 0xH4170
  %x = call half @llvm.exp.f16(half 1.0)
  ret half %x
}
declare half @llvm.exp.f16(half %x)

; CHECK: sqrt_call
define half @sqrt_call() {
; CHECK: ret half 0xH4000
  %x = call half @llvm.sqrt.f16(half 4.0)
  ret half %x
}
declare half @llvm.sqrt.f16(half %x)

; CHECK: floor_call
define half @floor_call() {
; CHECK: ret half 0xH4000
  %x = call half @llvm.floor.f16(half 2.5)
  ret half %x
}
declare half @llvm.floor.f16(half %x)

; CHECK: pow_call
define half @pow_call() {
; CHECK: ret half 0xH4400
  %x = call half @llvm.pow.f16(half 2.0, half 2.0)
  ret half %x
}
declare half @llvm.pow.f16(half %x, half %y)

