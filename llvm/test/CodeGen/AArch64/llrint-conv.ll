; RUN: llc < %s -mtriple=aarch64 -mattr=+neon | FileCheck %s

; CHECK-LABEL: testmsws:
; CHECK:       bl      llrintf
define i32 @testmsws(float %x) {
entry:
  %0 = tail call i64 @llvm.llrint.f32(float %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsxs:
; CHECK:       b       llrintf
define i64 @testmsxs(float %x) {
entry:
  %0 = tail call i64 @llvm.llrint.f32(float %x)
  ret i64 %0
}

; CHECK-LABEL: testmswd:
; CHECK:       bl      llrint
define i32 @testmswd(double %x) {
entry:
  %0 = tail call i64 @llvm.llrint.f64(double %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsxd:
; CHECK:       b       llrint
define i64 @testmsxd(double %x) {
entry:
  %0 = tail call i64 @llvm.llrint.f64(double %x)
  ret i64 %0
}

; CHECK-LABEL: testmswl:
; CHECK:       bl      llrintl
define i32 @testmswl(fp128 %x) {
entry:
  %0 = tail call i64 @llvm.llrint.f128(fp128 %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsll:
; CHECK:       b       llrintl
define i64 @testmsll(fp128 %x) {
entry:
  %0 = tail call i64 @llvm.llrint.f128(fp128 %x)
  ret i64 %0
}

declare i64 @llvm.llrint.f32(float) nounwind readnone
declare i64 @llvm.llrint.f64(double) nounwind readnone
declare i64 @llvm.llrint.f128(fp128) nounwind readnone
