; RUN: llc < %s -mtriple=i686-linux-pc -mcpu=corei7 | FileCheck %s

define void @test_convert_float2_ulong2(<2 x i64>* nocapture %src, <2 x float>* nocapture %dest) noinline {
L.entry:
  %0 = getelementptr <2 x i64>, <2 x i64>* %src, i32 10
  %1 = load <2 x i64>, <2 x i64>* %0, align 16
  %2 = uitofp <2 x i64> %1 to <2 x float>
  %3 = getelementptr <2 x float>, <2 x float>* %dest, i32 10
  store <2 x float> %2, <2 x float>* %3, align 8
  ret void
}

; CHECK: test_convert_float2_ulong2
; CHECK-NOT: cvtpd2ps
; CHECK: ret
