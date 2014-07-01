; RUN: llc -mtriple=i386-linux-gnu %s -o - | FileCheck %s

define i64 @test_add(i64* %addr, i64 %inc) {
; CHECK-LABEL: test_add:
; CHECK: calll __sync_fetch_and_add_8
  %old = atomicrmw add i64* %addr, i64 %inc seq_cst
  ret i64 %old
}

define i64 @test_sub(i64* %addr, i64 %inc) {
; CHECK-LABEL: test_sub:
; CHECK: calll __sync_fetch_and_sub_8
  %old = atomicrmw sub i64* %addr, i64 %inc seq_cst
  ret i64 %old
}

define i64 @test_and(i64* %andr, i64 %inc) {
; CHECK-LABEL: test_and:
; CHECK: calll __sync_fetch_and_and_8
  %old = atomicrmw and i64* %andr, i64 %inc seq_cst
  ret i64 %old
}

define i64 @test_or(i64* %orr, i64 %inc) {
; CHECK-LABEL: test_or:
; CHECK: calll __sync_fetch_and_or_8
  %old = atomicrmw or i64* %orr, i64 %inc seq_cst
  ret i64 %old
}

define i64 @test_xor(i64* %xorr, i64 %inc) {
; CHECK-LABEL: test_xor:
; CHECK: calll __sync_fetch_and_xor_8
  %old = atomicrmw xor i64* %xorr, i64 %inc seq_cst
  ret i64 %old
}

define i64 @test_nand(i64* %nandr, i64 %inc) {
; CHECK-LABEL: test_nand:
; CHECK: calll __sync_fetch_and_nand_8
  %old = atomicrmw nand i64* %nandr, i64 %inc seq_cst
  ret i64 %old
}
