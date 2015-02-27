; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=-f16c | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LIBCALL
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+f16c | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-F16C

define void @test_load_store(half* %in, half* %out) {
; CHECK-LABEL: test_load_store:
; CHECK: movw (%rdi), [[TMP:%[a-z0-9]+]]
; CHECK: movw [[TMP]], (%rsi)
  %val = load half, half* %in
  store half %val, half* %out
  ret void
}

define i16 @test_bitcast_from_half(half* %addr) {
; CHECK-LABEL: test_bitcast_from_half:
; CHECK: movzwl (%rdi), %eax
  %val = load half, half* %addr
  %val_int = bitcast half %val to i16
  ret i16 %val_int
}

define void @test_bitcast_to_half(half* %addr, i16 %in) {
; CHECK-LABEL: test_bitcast_to_half:
; CHECK: movw %si, (%rdi)
  %val_fp = bitcast i16 %in to half
  store half %val_fp, half* %addr
  ret void
}

define float @test_extend32(half* %addr) {
; CHECK-LABEL: test_extend32:

; CHECK-LIBCALL: jmp __gnu_h2f_ieee
; CHECK-FP16: vcvtph2ps
  %val16 = load half, half* %addr
  %val32 = fpext half %val16 to float
  ret float %val32
}

define double @test_extend64(half* %addr) {
; CHECK-LABEL: test_extend64:

; CHECK-LIBCALL: callq __gnu_h2f_ieee
; CHECK-LIBCALL: cvtss2sd
; CHECK-FP16: vcvtph2ps
; CHECK-FP16: vcvtss2sd
  %val16 = load half, half* %addr
  %val32 = fpext half %val16 to double
  ret double %val32
}

define void @test_trunc32(float %in, half* %addr) {
; CHECK-LABEL: test_trunc32:

; CHECK-LIBCALL: callq __gnu_f2h_ieee
; CHECK-FP16: vcvtps2ph
  %val16 = fptrunc float %in to half
  store half %val16, half* %addr
  ret void
}

define void @test_trunc64(double %in, half* %addr) {
; CHECK-LABEL: test_trunc64:

; CHECK-LIBCALL: callq __truncdfhf2
; CHECK-FP16: callq __truncdfhf2
  %val16 = fptrunc double %in to half
  store half %val16, half* %addr
  ret void
}
