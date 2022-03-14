;RUN: llc <%s -mattr=-neon  -mattr=-fp-armv8  | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

@t = common global i32 0, align 4
@x = common global i32 0, align 4

define void @foo() {
entry:
;CHECK-LABEL: foo:
;CHECK: __floatsisf
  %0 = load i32, i32* @x, align 4
  %conv = sitofp i32 %0 to float
  store float %conv, float* bitcast (i32* @t to float*), align 4
  ret void
}
