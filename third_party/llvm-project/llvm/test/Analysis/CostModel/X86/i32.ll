; RUN: opt < %s  -cost-model -analyze -mtriple=i386 -mcpu=corei7-avx | FileCheck %s

;CHECK: cost of 0 {{.*}} ret
define i32 @no_info(i32 %arg) {
  %e = add i64 undef, undef
  ret i32 undef
}
