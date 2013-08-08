; RUN: opt < %s -inline -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

define i32 @noattr_callee(i32 %i) {
  ret i32 %i
}

define i32 @sanitize_address_callee(i32 %i) sanitize_address {
  ret i32 %i
}

define i32 @sanitize_thread_callee(i32 %i) sanitize_thread {
  ret i32 %i
}

define i32 @sanitize_memory_callee(i32 %i) sanitize_memory {
  ret i32 %i
}

define i32 @alwaysinline_callee(i32 %i) alwaysinline {
  ret i32 %i
}

define i32 @alwaysinline_sanitize_address_callee(i32 %i) alwaysinline sanitize_address {
  ret i32 %i
}

define i32 @alwaysinline_sanitize_thread_callee(i32 %i) alwaysinline sanitize_thread {
  ret i32 %i
}

define i32 @alwaysinline_sanitize_memory_callee(i32 %i) alwaysinline sanitize_memory {
  ret i32 %i
}


; Check that:
;  * noattr callee is inlined into noattr caller,
;  * sanitize_(address|memory|thread) callee is not inlined into noattr caller,
;  * alwaysinline callee is always inlined no matter what sanitize_* attributes are present.

define i32 @test_no_sanitize_address(i32 %arg) {
  %x1 = call i32 @noattr_callee(i32 %arg)
  %x2 = call i32 @sanitize_address_callee(i32 %x1)
  %x3 = call i32 @alwaysinline_callee(i32 %x2)
  %x4 = call i32 @alwaysinline_sanitize_address_callee(i32 %x3)
  ret i32 %x4
; CHECK-LABEL: @test_no_sanitize_address(
; CHECK-NEXT: @sanitize_address_callee
; CHECK-NEXT: ret i32
}

define i32 @test_no_sanitize_memory(i32 %arg) {
  %x1 = call i32 @noattr_callee(i32 %arg)
  %x2 = call i32 @sanitize_memory_callee(i32 %x1)
  %x3 = call i32 @alwaysinline_callee(i32 %x2)
  %x4 = call i32 @alwaysinline_sanitize_memory_callee(i32 %x3)
  ret i32 %x4
; CHECK-LABEL: @test_no_sanitize_memory(
; CHECK-NEXT: @sanitize_memory_callee
; CHECK-NEXT: ret i32
}

define i32 @test_no_sanitize_thread(i32 %arg) {
  %x1 = call i32 @noattr_callee(i32 %arg)
  %x2 = call i32 @sanitize_thread_callee(i32 %x1)
  %x3 = call i32 @alwaysinline_callee(i32 %x2)
  %x4 = call i32 @alwaysinline_sanitize_thread_callee(i32 %x3)
  ret i32 %x4
; CHECK-LABEL: @test_no_sanitize_thread(
; CHECK-NEXT: @sanitize_thread_callee
; CHECK-NEXT: ret i32
}


; Check that:
;  * noattr callee is not inlined into sanitize_(address|memory|thread) caller,
;  * sanitize_(address|memory|thread) callee is inlined into the caller with the same attribute,
;  * alwaysinline callee is always inlined no matter what sanitize_* attributes are present.

define i32 @test_sanitize_address(i32 %arg) sanitize_address {
  %x1 = call i32 @noattr_callee(i32 %arg)
  %x2 = call i32 @sanitize_address_callee(i32 %x1)
  %x3 = call i32 @alwaysinline_callee(i32 %x2)
  %x4 = call i32 @alwaysinline_sanitize_address_callee(i32 %x3)
  ret i32 %x4
; CHECK-LABEL: @test_sanitize_address(
; CHECK-NEXT: @noattr_callee
; CHECK-NEXT: ret i32
}

define i32 @test_sanitize_memory(i32 %arg) sanitize_memory {
  %x1 = call i32 @noattr_callee(i32 %arg)
  %x2 = call i32 @sanitize_memory_callee(i32 %x1)
  %x3 = call i32 @alwaysinline_callee(i32 %x2)
  %x4 = call i32 @alwaysinline_sanitize_memory_callee(i32 %x3)
  ret i32 %x4
; CHECK-LABEL: @test_sanitize_memory(
; CHECK-NEXT: @noattr_callee
; CHECK-NEXT: ret i32
}

define i32 @test_sanitize_thread(i32 %arg) sanitize_thread {
  %x1 = call i32 @noattr_callee(i32 %arg)
  %x2 = call i32 @sanitize_thread_callee(i32 %x1)
  %x3 = call i32 @alwaysinline_callee(i32 %x2)
  %x4 = call i32 @alwaysinline_sanitize_thread_callee(i32 %x3)
  ret i32 %x4
; CHECK-LABEL: @test_sanitize_thread(
; CHECK-NEXT: @noattr_callee
; CHECK-NEXT: ret i32
}
