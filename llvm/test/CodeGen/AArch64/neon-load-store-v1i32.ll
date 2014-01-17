; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

; Test load/store of v1i8, v1i16, v1i32 types can be selected correctly
define void @load.store.v1i8(<1 x i8>* %ptr, <1 x i8>* %ptr2) {
; CHECK-LABEL: load.store.v1i8:
; CHECK: ldr b{{[0-9]+}}, [x{{[0-9]+|sp}}]
; CHECK: str b{{[0-9]+}}, [x{{[0-9]+|sp}}]
  %a = load <1 x i8>* %ptr
  store <1 x i8> %a, <1 x i8>* %ptr2
  ret void
}

define void @load.store.v1i16(<1 x i16>* %ptr, <1 x i16>* %ptr2) {
; CHECK-LABEL: load.store.v1i16:
; CHECK: ldr h{{[0-9]+}}, [x{{[0-9]+|sp}}]
; CHECK: str h{{[0-9]+}}, [x{{[0-9]+|sp}}]
  %a = load <1 x i16>* %ptr
  store <1 x i16> %a, <1 x i16>* %ptr2
  ret void
}

define void @load.store.v1i32(<1 x i32>* %ptr, <1 x i32>* %ptr2) {
; CHECK-LABEL: load.store.v1i32:
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+|sp}}]
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+|sp}}]
  %a = load <1 x i32>* %ptr
  store <1 x i32> %a, <1 x i32>* %ptr2
  ret void
}
