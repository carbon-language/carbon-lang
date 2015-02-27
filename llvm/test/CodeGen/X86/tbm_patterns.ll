; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+tbm < %s | FileCheck %s

define i32 @test_x86_tbm_bextri_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u32:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %0 = lshr i32 %a, 4
  %1 = and i32 %0, 4095
  ret i32 %1
}

define i32 @test_x86_tbm_bextri_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %0 = load i32, i32* %a
  %1 = lshr i32 %0, 4
  %2 = and i32 %1, 4095
  ret i32 %2
}

define i64 @test_x86_tbm_bextri_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u64:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %0 = lshr i64 %a, 4
  %1 = and i64 %0, 4095
  ret i64 %1
}

define i64 @test_x86_tbm_bextri_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %0 = load i64, i64* %a
  %1 = lshr i64 %0, 4
  %2 = and i64 %1, 4095
  ret i64 %2
}

define i32 @test_x86_tbm_blcfill_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcfill_u32:
  ; CHECK-NOT: mov
  ; CHECK: blcfill %
  %0 = add i32 %a, 1
  %1 = and i32 %0, %a
  ret i32 %1
}

define i64 @test_x86_tbm_blcfill_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcfill_u64:
  ; CHECK-NOT: mov
  ; CHECK: blcfill %
  %0 = add i64 %a, 1
  %1 = and i64 %0, %a
  ret i64 %1
}

define i32 @test_x86_tbm_blci_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blci_u32:
  ; CHECK-NOT: mov
  ; CHECK: blci %
  %0 = add i32 1, %a
  %1 = xor i32 %0, -1
  %2 = or i32 %1, %a
  ret i32 %2
}

define i64 @test_x86_tbm_blci_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blci_u64:
  ; CHECK-NOT: mov
  ; CHECK: blci %
  %0 = add i64 1, %a
  %1 = xor i64 %0, -1
  %2 = or i64 %1, %a
  ret i64 %2
}

define i32 @test_x86_tbm_blci_u32_b(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blci_u32_b:
  ; CHECK-NOT: mov
  ; CHECK: blci %
  %0 = sub i32 -2, %a
  %1 = or i32 %0, %a
  ret i32 %1
}

define i64 @test_x86_tbm_blci_u64_b(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blci_u64_b:
  ; CHECK-NOT: mov
  ; CHECK: blci %
  %0 = sub i64 -2, %a
  %1 = or i64 %0, %a
  ret i64 %1
}

define i32 @test_x86_tbm_blcic_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcic_u32:
  ; CHECK-NOT: mov
  ; CHECK: blcic %
  %0 = xor i32 %a, -1
  %1 = add i32 %a, 1
  %2 = and i32 %1, %0
  ret i32 %2
}

define i64 @test_x86_tbm_blcic_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcic_u64:
  ; CHECK-NOT: mov
  ; CHECK: blcic %
  %0 = xor i64 %a, -1
  %1 = add i64 %a, 1
  %2 = and i64 %1, %0
  ret i64 %2
}

define i32 @test_x86_tbm_blcmsk_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcmsk_u32:
  ; CHECK-NOT: mov
  ; CHECK: blcmsk %
  %0 = add i32 %a, 1
  %1 = xor i32 %0, %a
  ret i32 %1
}

define i64 @test_x86_tbm_blcmsk_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcmsk_u64:
  ; CHECK-NOT: mov
  ; CHECK: blcmsk %
  %0 = add i64 %a, 1
  %1 = xor i64 %0, %a
  ret i64 %1
}

define i32 @test_x86_tbm_blcs_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcs_u32:
  ; CHECK-NOT: mov
  ; CHECK: blcs %
  %0 = add i32 %a, 1
  %1 = or i32 %0, %a
  ret i32 %1
}

define i64 @test_x86_tbm_blcs_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcs_u64:
  ; CHECK-NOT: mov
  ; CHECK: blcs %
  %0 = add i64 %a, 1
  %1 = or i64 %0, %a
  ret i64 %1
}

define i32 @test_x86_tbm_blsfill_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsfill_u32:
  ; CHECK-NOT: mov
  ; CHECK: blsfill %
  %0 = add i32 %a, -1
  %1 = or i32 %0, %a
  ret i32 %1
}

define i64 @test_x86_tbm_blsfill_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsfill_u64:
  ; CHECK-NOT: mov
  ; CHECK: blsfill %
  %0 = add i64 %a, -1
  %1 = or i64 %0, %a
  ret i64 %1
}

define i32 @test_x86_tbm_blsic_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsic_u32:
  ; CHECK-NOT: mov
  ; CHECK: blsic %
  %0 = xor i32 %a, -1
  %1 = add i32 %a, -1
  %2 = or i32 %0, %1
  ret i32 %2
}

define i64 @test_x86_tbm_blsic_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsic_u64:
  ; CHECK-NOT: mov
  ; CHECK: blsic %
  %0 = xor i64 %a, -1
  %1 = add i64 %a, -1
  %2 = or i64 %0, %1
  ret i64 %2
}

define i32 @test_x86_tbm_t1mskc_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_t1mskc_u32:
  ; CHECK-NOT: mov
  ; CHECK: t1mskc %
  %0 = xor i32 %a, -1
  %1 = add i32 %a, 1
  %2 = or i32 %0, %1
  ret i32 %2
}

define i64 @Ttest_x86_tbm_t1mskc_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_t1mskc_u64:
  ; CHECK-NOT: mov
  ; CHECK: t1mskc %
  %0 = xor i64 %a, -1
  %1 = add i64 %a, 1
  %2 = or i64 %0, %1
  ret i64 %2
}

define i32 @test_x86_tbm_tzmsk_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_tzmsk_u32:
  ; CHECK-NOT: mov
  ; CHECK: tzmsk %
  %0 = xor i32 %a, -1
  %1 = add i32 %a, -1
  %2 = and i32 %0, %1
  ret i32 %2
}

define i64 @test_x86_tbm_tzmsk_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_tzmsk_u64:
  ; CHECK-NOT: mov
  ; CHECK: tzmsk %
  %0 = xor i64 %a, -1
  %1 = add i64 %a, -1
  %2 = and i64 %0, %1
  ret i64 %2
}
