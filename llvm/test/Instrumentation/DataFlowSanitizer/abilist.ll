; RUN: opt < %s -dfsan -dfsan-args-abi -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; CHECK: i32 @discard(i32 %a, i32 %b)
define i32 @discard(i32 %a, i32 %b) {
  ret i32 0
}

; CHECK: i32 @functional(i32 %a, i32 %b)
define i32 @functional(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

declare void @custom1(i32 %a, i32 %b)

declare i32 @custom2(i32 %a, i32 %b)

; CHECK: @f
define void @f() {
  ; CHECK: %[[LABELRETURN:.*]] = alloca i16

  ; CHECK: call void @__dfsw_custom1(i32 1, i32 2, i16 0, i16 0)
  call void @custom1(i32 1, i32 2)

  ; CHECK: call i32 @__dfsw_custom2(i32 1, i32 2, i16 0, i16 0, i16* %[[LABELRETURN]])
  call i32 @custom2(i32 1, i32 2)

  ret void
}

; CHECK: define linkonce_odr { i32, i16 } @"dfsw$custom2"(i32, i32, i16, i16)
; CHECK: %[[LABELRETURN2:.*]] = alloca i16
; CHECK: %[[RV:.*]] = call i32 @__dfsw_custom2
; CHECK: %[[RVSHADOW:.*]] = load i16* %[[LABELRETURN2]]
; CHECK: insertvalue {{.*}}[[RV]], 0
; CHECK: insertvalue {{.*}}[[RVSHADOW]], 1
; CHECK: ret { i32, i16 }

; CHECK: @g
define i32 (i32, i32)* @g() {
  ; CHECK: ret {{.*}} @"dfsw$custom2"
  ret i32 (i32, i32)* @custom2
}

; CHECK: declare void @__dfsw_custom1(i32, i32, i16, i16)
; CHECK: declare i32 @__dfsw_custom2(i32, i32, i16, i16, i16*)
