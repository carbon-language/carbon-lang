; RUN: opt < %s -dfsan -dfsan-args-abi -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

; CHECK: i32 @discard(i32 %a, i32 %b)
define i32 @discard(i32 %a, i32 %b) {
  ret i32 0
}

; CHECK: i32 @functional(i32 %a, i32 %b)
define i32 @functional(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

; CHECK: define i32 (i32, i32)* @discardg(i32 %0)
; CHECK: %[[CALL:.*]] = call { i32 (i32, i32)*, i[[#SBITS]] } @g.dfsan(i32 %0, i[[#SBITS]] 0)
; CHECK: %[[XVAL:.*]] = extractvalue { i32 (i32, i32)*, i[[#SBITS]] } %[[CALL]], 0
; CHECK: ret {{.*}} %[[XVAL]]
@discardg = alias i32 (i32, i32)* (i32), i32 (i32, i32)* (i32)* @g

declare void @custom1(i32 %a, i32 %b)

; CHECK: define linkonce_odr { i32, i[[#SBITS]] } @"dfsw$custom2"(i32 %0, i32 %1, i[[#SBITS]] %2, i[[#SBITS]] %3)
; CHECK: %[[LABELRETURN2:.*]] = alloca i[[#SBITS]]
; CHECK: %[[RV:.*]] = call i32 @__dfsw_custom2
; CHECK: %[[RVSHADOW:.*]] = load i[[#SBITS]], i[[#SBITS]]* %[[LABELRETURN2]]
; CHECK: insertvalue {{.*}}[[RV]], 0
; CHECK: insertvalue {{.*}}[[RVSHADOW]], 1
; CHECK: ret { i32, i[[#SBITS]] }
declare i32 @custom2(i32 %a, i32 %b)

; CHECK: define linkonce_odr void @"dfsw$custom3"(i32 %0, i[[#SBITS]] %1, i[[#SBITS]]* %2, ...)
; CHECK: call void @__dfsan_vararg_wrapper(i8*
; CHECK: unreachable
declare void @custom3(i32 %a, ...)

declare i32 @custom4(i32 %a, ...)

declare void @customcb(i32 (i32)* %cb)

declare i32 @cb(i32)

; CHECK: @f.dfsan
define void @f(i32 %x) {
  ; CHECK: %[[LABELVA2:.*]] = alloca [2 x i[[#SBITS]]]
  ; CHECK: %[[LABELVA1:.*]] = alloca [2 x i[[#SBITS]]]
  ; CHECK: %[[LABELRETURN:.*]] = alloca i[[#SBITS]]

  ; CHECK: call void @__dfsw_custom1(i32 1, i32 2, i[[#SBITS]] zeroext 0, i[[#SBITS]] zeroext 0)
  call void @custom1(i32 1, i32 2)

  ; CHECK: call i32 @__dfsw_custom2(i32 1, i32 2, i[[#SBITS]] zeroext 0, i[[#SBITS]] zeroext 0, i[[#SBITS]]* %[[LABELRETURN]])
  call i32 @custom2(i32 1, i32 2)

  ; CHECK: call void @__dfsw_customcb({{.*}} @"dfst0$customcb", i8* bitcast ({{.*}} @cb.dfsan to i8*), i[[#SBITS]] zeroext 0)
  call void @customcb(i32 (i32)* @cb)

  ; CHECK: %[[LABELVA1_0:.*]] = getelementptr inbounds [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* %[[LABELVA1]], i32 0, i32 0
  ; CHECK: store i[[#SBITS]] 0, i[[#SBITS]]* %[[LABELVA1_0]]
  ; CHECK: %[[LABELVA1_1:.*]] = getelementptr inbounds [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* %[[LABELVA1]], i32 0, i32 1
  ; CHECK: store i[[#SBITS]] %{{.*}}, i[[#SBITS]]* %[[LABELVA1_1]]
  ; CHECK: %[[LABELVA1_0A:.*]] = getelementptr inbounds [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* %[[LABELVA1]], i32 0, i32 0
  ; CHECK: call void (i32, i[[#SBITS]], i[[#SBITS]]*, ...) @__dfsw_custom3(i32 1, i[[#SBITS]] zeroext 0, i[[#SBITS]]* %[[LABELVA1_0A]], i32 2, i32 %{{.*}})
  call void (i32, ...) @custom3(i32 1, i32 2, i32 %x)

  ; CHECK: %[[LABELVA2_0:.*]] = getelementptr inbounds [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* %[[LABELVA2]], i32 0, i32 0
  ; CHECK: %[[LABELVA2_0A:.*]] = getelementptr inbounds [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* %[[LABELVA2]], i32 0, i32 0
  ; CHECK: call i32 (i32, i[[#SBITS]], i[[#SBITS]]*, i[[#SBITS]]*, ...) @__dfsw_custom4(i32 1, i[[#SBITS]] zeroext 0, i[[#SBITS]]* %[[LABELVA2_0A]], i[[#SBITS]]* %[[LABELRETURN]], i32 2, i32 3)
  call i32 (i32, ...) @custom4(i32 1, i32 2, i32 3)

  ret void
}

; CHECK: @g.dfsan
define i32 (i32, i32)* @g(i32) {
  ; CHECK: ret {{.*}} @"dfsw$custom2"
  ret i32 (i32, i32)* @custom2
}

; CHECK: define { i32, i[[#SBITS]] } @adiscard.dfsan(i32 %0, i32 %1, i[[#SBITS]] %2, i[[#SBITS]] %3)
; CHECK: %[[CALL:.*]] = call i32 @discard(i32 %0, i32 %1)
; CHECK: %[[IVAL0:.*]] = insertvalue { i32, i[[#SBITS]] } undef, i32 %[[CALL]], 0
; CHECK: %[[IVAL1:.*]] = insertvalue { i32, i[[#SBITS]] } %[[IVAL0]], i[[#SBITS]] 0, 1
; CHECK: ret { i32, i[[#SBITS]] } %[[IVAL1]]
@adiscard = alias i32 (i32, i32), i32 (i32, i32)* @discard

; CHECK: declare void @__dfsw_custom1(i32, i32, i[[#SBITS]], i[[#SBITS]])
; CHECK: declare i32 @__dfsw_custom2(i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)

; CHECK-LABEL: define linkonce_odr i32 @"dfst0$customcb"
; CHECK-SAME: (i32 (i32)* %0, i32 %1, i[[#SBITS]] %2, i[[#SBITS]]* %3)
; CHECK: %[[BC:.*]] = bitcast i32 (i32)* %0 to { i32, i[[#SBITS]] } (i32, i[[#SBITS]])*
; CHECK: %[[CALL:.*]] = call { i32, i[[#SBITS]] } %[[BC]](i32 %1, i[[#SBITS]] %2)
; CHECK: %[[XVAL0:.*]] = extractvalue { i32, i[[#SBITS]] } %[[CALL]], 0
; CHECK: %[[XVAL1:.*]] = extractvalue { i32, i[[#SBITS]] } %[[CALL]], 1
; CHECK: store i[[#SBITS]] %[[XVAL1]], i[[#SBITS]]* %3
; CHECK: ret i32 %[[XVAL0]]

; CHECK: declare void @__dfsw_custom3(i32, i[[#SBITS]], i[[#SBITS]]*, ...)
; CHECK: declare i32 @__dfsw_custom4(i32, i[[#SBITS]], i[[#SBITS]]*, i[[#SBITS]]*, ...)
