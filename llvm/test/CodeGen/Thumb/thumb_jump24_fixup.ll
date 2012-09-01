; RUN: llc -mtriple thumbv7-none-linux-gnueabi -mcpu=cortex-a8 -march=thumb -mattr=thumb2 -filetype=obj -o - < %s | llvm-objdump -r - | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:32-n32-S64"
target triple = "thumbv7-none-linux-gnueabi"

define i32 @test_fixup_t2_uncondbranch() {
b0:
  invoke void @__cxa_throw(i8* null, i8* null, i8* null) noreturn
    to label %unreachable unwind label %lpad

; CHECK: {{[0-9]+}} R_ARM_THM_JUMP24 __cxa_throw

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) cleanup
  ret i32 0

unreachable:
  unreachable
}

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(i8*, i8*, i8*)
