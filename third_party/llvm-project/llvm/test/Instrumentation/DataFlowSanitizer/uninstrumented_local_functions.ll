; RUN: opt < %s -dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define internal i8 @uninstrumented_internal_fun(i8 %in) {
  ret i8 %in
}

define i8 @call_uninstrumented_internal_fun(i8 %in) {
  %call = call i8 @uninstrumented_internal_fun(i8 %in)
  ret i8 %call
}
; CHECK: define internal i8 @"dfsw$uninstrumented_internal_fun"

define private i8 @uninstrumented_private_fun(i8 %in) {
  ret i8 %in
}

define i8 @call_uninstrumented_private_fun(i8 %in) {
  %call = call i8 @uninstrumented_private_fun(i8 %in)
  ret i8 %call
}
; CHECK: define private i8 @"dfsw$uninstrumented_private_fun"
