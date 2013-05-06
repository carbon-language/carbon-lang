; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare float @llvm.fma.f32(float %f1, float %f2, float %f3)

define float @f1(float %f1, float %f2, float %acc) {
; CHECK: f1:
; CHECK: msebr %f4, %f0, %f2
; CHECK: ler %f0, %f4
; CHECK: br %r14
  %negacc = fsub float -0.0, %acc
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %negacc)
  ret float %res
}

define float @f2(float %f1, float *%ptr, float %acc) {
; CHECK: f2:
; CHECK: mseb %f2, %f0, 0(%r2)
; CHECK: ler %f0, %f2
; CHECK: br %r14
  %f2 = load float *%ptr
  %negacc = fsub float -0.0, %acc
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %negacc)
  ret float %res
}

define float @f3(float %f1, float *%base, float %acc) {
; CHECK: f3:
; CHECK: mseb %f2, %f0, 4092(%r2)
; CHECK: ler %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1023
  %f2 = load float *%ptr
  %negacc = fsub float -0.0, %acc
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %negacc)
  ret float %res
}

define float @f4(float %f1, float *%base, float %acc) {
; The important thing here is that we don't generate an out-of-range
; displacement.  Other sequences besides this one would be OK.
;
; CHECK: f4:
; CHECK: aghi %r2, 4096
; CHECK: mseb %f2, %f0, 0(%r2)
; CHECK: ler %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1024
  %f2 = load float *%ptr
  %negacc = fsub float -0.0, %acc
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %negacc)
  ret float %res
}

define float @f5(float %f1, float *%base, float %acc) {
; Here too the important thing is that we don't generate an out-of-range
; displacement.  Other sequences besides this one would be OK.
;
; CHECK: f5:
; CHECK: aghi %r2, -4
; CHECK: mseb %f2, %f0, 0(%r2)
; CHECK: ler %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 -1
  %f2 = load float *%ptr
  %negacc = fsub float -0.0, %acc
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %negacc)
  ret float %res
}

define float @f6(float %f1, float *%base, i64 %index, float %acc) {
; CHECK: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: mseb %f2, %f0, 0(%r1,%r2)
; CHECK: ler %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 %index
  %f2 = load float *%ptr
  %negacc = fsub float -0.0, %acc
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %negacc)
  ret float %res
}

define float @f7(float %f1, float *%base, i64 %index, float %acc) {
; CHECK: f7:
; CHECK: sllg %r1, %r3, 2
; CHECK: mseb %f2, %f0, 4092({{%r1,%r2|%r2,%r1}})
; CHECK: ler %f0, %f2
; CHECK: br %r14
  %index2 = add i64 %index, 1023
  %ptr = getelementptr float *%base, i64 %index2
  %f2 = load float *%ptr
  %negacc = fsub float -0.0, %acc
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %negacc)
  ret float %res
}

define float @f8(float %f1, float *%base, i64 %index, float %acc) {
; CHECK: f8:
; CHECK: sllg %r1, %r3, 2
; CHECK: lay %r1, 4096({{%r1,%r2|%r2,%r1}})
; CHECK: mseb %f2, %f0, 0(%r1)
; CHECK: ler %f0, %f2
; CHECK: br %r14
  %index2 = add i64 %index, 1024
  %ptr = getelementptr float *%base, i64 %index2
  %f2 = load float *%ptr
  %negacc = fsub float -0.0, %acc
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %negacc)
  ret float %res
}
