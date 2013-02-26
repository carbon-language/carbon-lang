; Test non-default shadow mapping scale and offset.
;
; RUN: opt < %s -asan -asan-mapping-scale=2 -asan-mapping-offset-log=0 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

; Test that ASan tells scale and offset to runtime.
; CHECK: @__asan_mapping_offset = linkonce_odr constant i64 0
; CHECK: @__asan_mapping_scale = linkonce_odr constant i64 2

define i32 @test_load(i32* %a) sanitize_address {
; CHECK: @test_load
; CHECK-NOT: load
; CHECK:   %[[LOAD_ADDR:[^ ]*]] = ptrtoint i32* %a to i64
; CHECK:   lshr i64 %[[LOAD_ADDR]], 2

; No need in shift for zero offset.
; CHECK-NOT:  or i64

; CHECK:   %[[LOAD_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[LOAD_SHADOW:[^ ]*]] = load i8* %[[LOAD_SHADOW_PTR]]
; CHECK:   icmp ne i8
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}

; No need in slow path for i32 and mapping scale equal to 2.
; CHECK-NOT:   and i64 %[[LOAD_ADDR]]
;
; The crash block reports the error.
; CHECK:   call void @__asan_report_load4(i64 %[[LOAD_ADDR]])
; CHECK:   unreachable
;
; The actual load.
; CHECK:   %tmp1 = load i32* %a
; CHECK:   ret i32 %tmp1

entry:
  %tmp1 = load i32* %a
  ret i32 %tmp1
}

