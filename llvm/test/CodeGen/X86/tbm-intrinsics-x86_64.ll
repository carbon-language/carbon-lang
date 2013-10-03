; RUN: llc -mtriple=x86_64-unknown-unknown -march=x86-64 -mattr=+tbm < %s | FileCheck %s

define i32 @test_x86_tbm_bextri_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u32:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %0 = tail call i32 @llvm.x86.tbm.bextri.u32(i32 %a, i32 2814)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.bextri.u32(i32, i32) nounwind readnone

define i32 @test_x86_tbm_bextri_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %tmp1 = load i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.bextri.u32(i32 %tmp1, i32 2814)
  ret i32 %0
}

define i64 @test_x86_tbm_bextri_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u64:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %0 = tail call i64 @llvm.x86.tbm.bextri.u64(i64 %a, i64 2814)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.bextri.u64(i64, i64) nounwind readnone

define i64 @test_x86_tbm_bextri_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEl: test_x86_tbm_bextri_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %tmp1 = load i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.bextri.u64(i64 %tmp1, i64 2814)
  ret i64 %0
}

define i32 @test_x86_tbm_blcfill_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcfill_u32:
  ; CHECK-NOT: mov
  ; CHECK: blcfill %
  %0 = tail call i32 @llvm.x86.tbm.blcfill.u32(i32 %a)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.blcfill.u32(i32) nounwind readnone

define i32 @test_x86_tbm_blcfill_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcfill_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: blcfill (%
  %tmp1 = load i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.blcfill.u32(i32 %tmp1)
  ret i32 %0
}

define i64 @test_x86_tbm_blcfill_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcfill_u64:
  ; CHECK-NOT: mov
  ; CHECK: blcfill %
  %0 = tail call i64 @llvm.x86.tbm.blcfill.u64(i64 %a)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.blcfill.u64(i64) nounwind readnone

define i64 @test_x86_tbm_blcfill_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcfill_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: blcfill (%
  %tmp1 = load i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.blcfill.u64(i64 %tmp1)
  ret i64 %0
}

define i32 @test_x86_tbm_blci_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blci_u32:
  ; CHECK-NOT: mov
  ; CHECK: blci %
  %0 = tail call i32 @llvm.x86.tbm.blci.u32(i32 %a)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.blci.u32(i32) nounwind readnone

define i32 @test_x86_tbm_blci_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blci_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: blci (%
  %tmp1 = load i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.blci.u32(i32 %tmp1)
  ret i32 %0
}

define i64 @test_x86_tbm_blci_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blci_u64:
  ; CHECK-NOT: mov
  ; CHECK: blci %
  %0 = tail call i64 @llvm.x86.tbm.blci.u64(i64 %a)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.blci.u64(i64) nounwind readnone

define i64 @test_x86_tbm_blci_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEl: test_x86_tbm_blci_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: blci (%
  %tmp1 = load i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.blci.u64(i64 %tmp1)
  ret i64 %0
}

define i32 @test_x86_tbm_blcic_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcic_u32:
  ; CHECK-NOT: mov
  ; CHECK: blcic %
  %0 = tail call i32 @llvm.x86.tbm.blcic.u32(i32 %a)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.blcic.u32(i32) nounwind readnone

define i32 @test_x86_tbm_blcic_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcic_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: blcic (%
  %tmp1 = load i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.blcic.u32(i32 %tmp1)
  ret i32 %0
}

define i64 @test_x86_tbm_blcic_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcic_u64:
  ; CHECK-NOT: mov
  ; CHECK: blcic %
  %0 = tail call i64 @llvm.x86.tbm.blcic.u64(i64 %a)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.blcic.u64(i64) nounwind readnone

define i64 @test_x86_tbm_blcic_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcic_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: blcic (%
  %tmp1 = load i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.blcic.u64(i64 %tmp1)
  ret i64 %0
}

define i32 @test_x86_tbm_blcmsk_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcmsk_u32:
  ; CHECK-NOT: mov
  ; CHECK: blcmsk %
  %0 = tail call i32 @llvm.x86.tbm.blcmsk.u32(i32 %a)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.blcmsk.u32(i32) nounwind readnone

define i32 @test_x86_tbm_blcmsk_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcmsk_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: blcmsk (%
  %tmp1 = load i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.blcmsk.u32(i32 %tmp1)
  ret i32 %0
}

define i64 @test_x86_tbm_blcmsk_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcmsk_u64:
  ; CHECK-NOT: mov
  ; CHECK: blcmsk %
  %0 = tail call i64 @llvm.x86.tbm.blcmsk.u64(i64 %a)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.blcmsk.u64(i64) nounwind readnone

define i64 @test_x86_tbm_blcmsk_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcmsk_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: blcmsk (%
  %tmp1 = load i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.blcmsk.u64(i64 %tmp1)
  ret i64 %0
}

define i32 @test_x86_tbm_blcs_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcs_u32:
  ; CHECK-NOT: mov
  ; CHECK: blcs %
  %0 = tail call i32 @llvm.x86.tbm.blcs.u32(i32 %a)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.blcs.u32(i32) nounwind readnone

define i32 @test_x86_tbm_blcs_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcs_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: blcs (%
  %tmp1 = load i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.blcs.u32(i32 %tmp1)
  ret i32 %0
}

define i64 @test_x86_tbm_blcs_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcs_u64:
  ; CHECK-NOT: mov
  ; CHECK: blcs %
  %0 = tail call i64 @llvm.x86.tbm.blcs.u64(i64 %a)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.blcs.u64(i64) nounwind readnone

define i64 @test_x86_tbm_blcs_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blcs_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: blcs (%
  %tmp1 = load i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.blcs.u64(i64 %tmp1)
  ret i64 %0
}

define i32 @test_x86_tbm_blsfill_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsfill_u32:
  ; CHECK-NOT: mov
  ; CHECK: blsfill %
  %0 = tail call i32 @llvm.x86.tbm.blsfill.u32(i32 %a)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.blsfill.u32(i32) nounwind readnone

define i32 @test_x86_tbm_blsfill_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsfill_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: blsfill (%
  %tmp1 = load i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.blsfill.u32(i32 %tmp1)
  ret i32 %0
}

define i64 @test_x86_tbm_blsfill_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsfill_u64:
  ; CHECK-NOT: mov
  ; CHECK: blsfill %
  %0 = tail call i64 @llvm.x86.tbm.blsfill.u64(i64 %a)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.blsfill.u64(i64) nounwind readnone

define i64 @test_x86_tbm_blsfill_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsfill_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: blsfill (%
  %tmp1 = load i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.blsfill.u64(i64 %tmp1)
  ret i64 %0
}

define i32 @test_x86_tbm_blsic_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsic_u32:
  ; CHECK-NOT: mov
  ; CHECK: blsic %
  %0 = tail call i32 @llvm.x86.tbm.blsic.u32(i32 %a)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.blsic.u32(i32) nounwind readnone

define i32 @test_x86_tbm_blsic_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsic_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: blsic (%
  %tmp1 = load i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.blsic.u32(i32 %tmp1)
  ret i32 %0
}

define i64 @test_x86_tbm_blsic_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsic_u64:
  ; CHECK-NOT: mov
  ; CHECK: blsic %
  %0 = tail call i64 @llvm.x86.tbm.blsic.u64(i64 %a)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.blsic.u64(i64) nounwind readnone

define i64 @test_x86_tbm_blsic_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_blsic_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: blsic (%
  %tmp1 = load i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.blsic.u64(i64 %tmp1)
  ret i64 %0
}

define i32 @test_x86_tbm_t1mskc_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_t1mskc_u32:
  ; CHECK-NOT: mov
  ; CHECK: t1mskc %
  %0 = tail call i32 @llvm.x86.tbm.t1mskc.u32(i32 %a)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.t1mskc.u32(i32) nounwind readnone

define i32 @test_x86_tbm_t1mskc_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_t1mskc_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: t1mskc (%
  %tmp1 = load i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.t1mskc.u32(i32 %tmp1)
  ret i32 %0
}

define i64 @test_x86_tbm_t1mskc_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_t1mskc_u64:
  ; CHECK-NOT: mov
  ; CHECK: t1mskc %
  %0 = tail call i64 @llvm.x86.tbm.t1mskc.u64(i64 %a)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.t1mskc.u64(i64) nounwind readnone

define i64 @test_x86_tbm_t1mskc_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_t1mskc_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: t1mskc (%
  %tmp1 = load i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.t1mskc.u64(i64 %tmp1)
  ret i64 %0
}

define i32 @test_x86_tbm_tzmsk_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_tzmsk_u32:
  ; CHECK-NOT: mov
  ; CHECK: tzmsk %
  %0 = tail call i32 @llvm.x86.tbm.tzmsk.u32(i32 %a)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.tzmsk.u32(i32) nounwind readnone

define i32 @test_x86_tbm_tzmsk_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_tzmsk_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: tzmsk (%
  %tmp1 = load i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.tzmsk.u32(i32 %tmp1)
  ret i32 %0
}

define i64 @test_x86_tbm_tzmsk_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_tzmsk_u64:
  ; CHECK-NOT: mov
  ; CHECK: tzmsk %
  %0 = tail call i64 @llvm.x86.tbm.tzmsk.u64(i64 %a)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.tzmsk.u64(i64) nounwind readnone

define i64 @test_x86_tbm_tzmsk_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEl: test_x86_tbm_tzmsk_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: tzmsk (%
  %tmp1 = load i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.tzmsk.u64(i64 %tmp1)
  ret i64 %0
}

