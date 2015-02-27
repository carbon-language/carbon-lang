; Test asan internal compiler flags:
;   -asan-instrument-allocas=1

; RUN: opt < %s -asan -asan-module -asan-instrument-allocas=1 -S | FileCheck %s --check-prefix=CHECK-ALLOCA
; RUN: opt < %s -asan -asan-module -asan-instrument-allocas=0 -S | FileCheck %s --check-prefix=CHECK-NOALLOCA
; RUN: opt < %s -asan -asan-module -S | FileCheck %s --check-prefix=CHECK-NOALLOCA
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %len) sanitize_address {
entry:
; CHECK-ALLOCA: store i32 -892679478
; CHECK-ALLOCA: store i32 -875836469
; CHECK-NOALLOCA-NOT: store i32 -892679478
; CHECK-NOALLOCA-NOT: store i32 -875836469
  %0 = alloca i32, align 4
  %1 = alloca i8*
  store volatile i32 %len, i32* %0, align 4
  %2 = load i32* %0, align 4
  %3 = zext i32 %2 to i64
  %4 = alloca i8, i64 %3, align 32
  store volatile i8 0, i8* %4
  ret void
}

