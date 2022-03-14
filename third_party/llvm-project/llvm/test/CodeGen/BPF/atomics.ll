; RUN: llc < %s -march=bpfel -verify-machineinstrs -show-mc-encoding | FileCheck %s
; RUN: llc < %s -march=bpfel -verify-machineinstrs -show-mc-encoding -mcpu=v3 | FileCheck --check-prefix=CHECK-V3 %s

; CHECK-LABEL: test_load_add_32
; CHECK: lock *(u32 *)(r1 + 0) += r2
; CHECK: encoding: [0xc3,0x21
; CHECK-V3: lock *(u32 *)(r1 + 0) += w2
; CHECK-V3: encoding: [0xc3,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
define void @test_load_add_32(i32* %p, i32 zeroext %v) {
entry:
  atomicrmw add i32* %p, i32 %v seq_cst
  ret void
}

; CHECK-LABEL: test_load_add_64
; CHECK: lock *(u64 *)(r1 + 0) += r2
; CHECK: encoding: [0xdb,0x21
; CHECK-V3: lock *(u64 *)(r1 + 0) += r2
; CHECK-V3: encoding: [0xdb,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
define void @test_load_add_64(i64* %p, i64 zeroext %v) {
entry:
  atomicrmw add i64* %p, i64 %v seq_cst
  ret void
}
