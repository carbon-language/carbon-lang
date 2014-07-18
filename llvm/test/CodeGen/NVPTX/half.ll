; RUN: llc < %s -march=nvptx | FileCheck %s

define void @test_load_store(half addrspace(1)* %in, half addrspace(1)* %out) {
; CHECK-LABEL: @test_load_store
; CHECK: ld.global.u16 [[TMP:%rs[0-9]+]], [{{%r[0-9]+}}]
; CHECK: st.global.u16 [{{%r[0-9]+}}], [[TMP]]
  %val = load half addrspace(1)* %in
  store half %val, half addrspace(1) * %out
  ret void
}

define void @test_bitcast_from_half(half addrspace(1)* %in, i16 addrspace(1)* %out) {
; CHECK-LABEL: @test_bitcast_from_half
; CHECK: ld.global.u16 [[TMP:%rs[0-9]+]], [{{%r[0-9]+}}]
; CHECK: st.global.u16 [{{%r[0-9]+}}], [[TMP]]
  %val = load half addrspace(1) * %in
  %val_int = bitcast half %val to i16
  store i16 %val_int, i16 addrspace(1)* %out
  ret void
}

define void @test_bitcast_to_half(half addrspace(1)* %out, i16 addrspace(1)* %in) {
; CHECK-LABEL: @test_bitcast_to_half
; CHECK: ld.global.u16 [[TMP:%rs[0-9]+]], [{{%r[0-9]+}}]
; CHECK: st.global.u16 [{{%r[0-9]+}}], [[TMP]]
  %val = load i16 addrspace(1)* %in
  %val_fp = bitcast i16 %val to half
  store half %val_fp, half addrspace(1)* %out
  ret void
}
