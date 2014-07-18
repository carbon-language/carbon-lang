; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=-f16c | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LIBCALL
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+f16c | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-F16C

define void @test_load_store(half* %in, half* %out) {
; CHECK-LABEL: test_load_store:
; CHECK: movw (%rdi), [[TMP:%[a-z0-9]+]]
; CHECK: movw [[TMP]], (%rsi)
  %val = load half* %in
  store half %val, half* %out
  ret void
}

define i16 @test_bitcast_from_half(half* %addr) {
; CHECK-LABEL: test_bitcast_from_half:
; CHECK: movzwl (%rdi), %eax
  %val = load half* %addr
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
