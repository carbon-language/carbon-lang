; RUN: llc < %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

define void @test_load_store(half* %in, half* %out) {
; CHECK-LABEL: test_load_store:
; CHECK: ldr [[TMP:h[0-9]+]], [x0]
; CHECK: str [[TMP]], [x1]
  %val = load half* %in
  store half %val, half* %out
  ret void
}

define i16 @test_bitcast_from_half(half* %addr) {
; CHECK-LABEL: test_bitcast_from_half:
; CHECK: ldrh w0, [x0]
  %val = load half* %addr
  %val_int = bitcast half %val to i16
  ret i16 %val_int
}

define void @test_bitcast_to_half(half* %addr, i16 %in) {
; CHECK-LABEL: test_bitcast_to_half:
; CHECK: strh w1, [x0]
  %val_fp = bitcast i16 %in to half
  store half %val_fp, half* %addr
  ret void
}

define float @test_extend32(half* %addr) {
; CHECK-LABEL: test_extend32:
; CHECK: fcvt {{s[0-9]+}}, {{h[0-9]+}}

  %val16 = load half* %addr
  %val32 = fpext half %val16 to float
  ret float %val32
}

define double @test_extend64(half* %addr) {
; CHECK-LABEL: test_extend64:
; CHECK: fcvt {{d[0-9]+}}, {{h[0-9]+}}

  %val16 = load half* %addr
  %val32 = fpext half %val16 to double
  ret double %val32
}

define void @test_trunc32(float %in, half* %addr) {
; CHECK-LABEL: test_trunc32:
; CHECK: fcvt {{h[0-9]+}}, {{s[0-9]+}}

  %val16 = fptrunc float %in to half
  store half %val16, half* %addr
  ret void
}

define void @test_trunc64(double %in, half* %addr) {
; CHECK-LABEL: test_trunc64:
; CHECK: fcvt {{h[0-9]+}}, {{d[0-9]+}}

  %val16 = fptrunc double %in to half
  store half %val16, half* %addr
  ret void
}
