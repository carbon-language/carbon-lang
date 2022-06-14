; RUN: llc < %s -debug-only=legalize-types 2>&1 | FileCheck %s --check-prefix=CHECK-LEGALIZATION
; RUN: llc < %s | FileCheck %s
; REQUIRES: asserts

target triple = "aarch64-unknown-linux-gnu"
attributes #0 = {"target-features"="+sve" uwtable}

declare <vscale x 2 x i64> @llvm.experimental.vector.insert.nxv2i64.v8i64(<vscale x 2 x i64>, <8 x i64>, i64)
declare <vscale x 2 x double> @llvm.experimental.vector.insert.nxv2f64.v8f64(<vscale x 2 x double>, <8 x double>, i64)

define <vscale x 2 x i64> @test_nxv2i64_v8i64(<vscale x 2 x i64> %a, <8 x i64> %b) #0 {
; CHECK-LEGALIZATION: Legally typed node: [[T1:t[0-9]+]]: nxv2i64 = insert_subvector {{t[0-9]+}}, {{t[0-9]+}}, Constant:i64<0>
; CHECK-LEGALIZATION: Legally typed node: [[T2:t[0-9]+]]: nxv2i64 = insert_subvector [[T1]], {{t[0-9]+}}, Constant:i64<2>
; CHECK-LEGALIZATION: Legally typed node: [[T3:t[0-9]+]]: nxv2i64 = insert_subvector [[T2]], {{t[0-9]+}}, Constant:i64<4>
; CHECK-LEGALIZATION: Legally typed node: [[T4:t[0-9]+]]: nxv2i64 = insert_subvector [[T3]], {{t[0-9]+}}, Constant:i64<6>

; CHECK-LABEL: test_nxv2i64_v8i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    str x29, [sp, #-16]! // 8-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    addvl sp, sp, #-3
; CHECK-NEXT:    .cfi_escape 0x0f, 0x0c, 0x8f, 0x00, 0x11, 0x10, 0x22, 0x11, 0x18, 0x92, 0x2e, 0x00, 0x1e, 0x22 // sp + 16 + 24 * VG
; CHECK-NEXT:    cntd x8
; CHECK-NEXT:    mov w9, #2
; CHECK-NEXT:    sub x8, x8, #2
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    cmp x8, #2
; CHECK-NEXT:    mov x10, sp
; CHECK-NEXT:    csel x9, x8, x9, lo
; CHECK-NEXT:    // kill: def $q1 killed $q1 def $z1
; CHECK-NEXT:    mov z0.d, p0/m, z1.d
; CHECK-NEXT:    lsl x9, x9, #3
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    st1d { z0.d }, p0, [sp]
; CHECK-NEXT:    cmp x8, #4
; CHECK-NEXT:    str q2, [x10, x9]
; CHECK-NEXT:    mov w9, #4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [sp]
; CHECK-NEXT:    csel x9, x8, x9, lo
; CHECK-NEXT:    lsl x9, x9, #3
; CHECK-NEXT:    addvl x10, sp, #1
; CHECK-NEXT:    cmp x8, #6
; CHECK-NEXT:    st1d { z0.d }, p0, [sp, #1, mul vl]
; CHECK-NEXT:    str q3, [x10, x9]
; CHECK-NEXT:    mov w9, #6
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [sp, #1, mul vl]
; CHECK-NEXT:    csel x8, x8, x9, lo
; CHECK-NEXT:    addvl x9, sp, #2
; CHECK-NEXT:    lsl x8, x8, #3
; CHECK-NEXT:    st1d { z0.d }, p0, [sp, #2, mul vl]
; CHECK-NEXT:    str q4, [x9, x8]
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [sp, #2, mul vl]
; CHECK-NEXT:    addvl sp, sp, #3
; CHECK-NEXT:    .cfi_def_cfa wsp, 16
; CHECK-NEXT:    ldr x29, [sp], #16 // 8-byte Folded Reload
; CHECK-NEXT:   .cfi_def_cfa_offset 0
; CHECK-NEXT:   .cfi_restore w29
; CHECK-NEXT:    ret




  %r = call <vscale x 2 x i64> @llvm.experimental.vector.insert.nxv2i64.v8i64(<vscale x 2 x i64> %a, <8 x i64> %b, i64 0)
  ret <vscale x 2 x i64> %r
}

define <vscale x 2 x double> @test_nxv2f64_v8f64(<vscale x 2 x double> %a, <8 x double> %b) #0 {
; CHECK-LEGALIZATION: Legally typed node: [[T1:t[0-9]+]]: nxv2f64 = insert_subvector {{t[0-9]+}}, {{t[0-9]+}}, Constant:i64<0>
; CHECK-LEGALIZATION: Legally typed node: [[T2:t[0-9]+]]: nxv2f64 = insert_subvector [[T1]], {{t[0-9]+}}, Constant:i64<2>
; CHECK-LEGALIZATION: Legally typed node: [[T3:t[0-9]+]]: nxv2f64 = insert_subvector [[T2]], {{t[0-9]+}}, Constant:i64<4>
; CHECK-LEGALIZATION: Legally typed node: [[T4:t[0-9]+]]: nxv2f64 = insert_subvector [[T3]], {{t[0-9]+}}, Constant:i64<6>

; CHECK-LABEL: test_nxv2f64_v8f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    str x29, [sp, #-16]! // 8-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    addvl sp, sp, #-3
; CHECK-NEXT:    .cfi_escape 0x0f, 0x0c, 0x8f, 0x00, 0x11, 0x10, 0x22, 0x11, 0x18, 0x92, 0x2e, 0x00, 0x1e, 0x22 // sp + 16 + 24 * VG
; CHECK-NEXT:    cntd x8
; CHECK-NEXT:    mov w9, #2
; CHECK-NEXT:    sub x8, x8, #2
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    cmp x8, #2
; CHECK-NEXT:    mov x10, sp
; CHECK-NEXT:    csel x9, x8, x9, lo
; CHECK-NEXT:    // kill: def $q1 killed $q1 def $z1
; CHECK-NEXT:    mov z0.d, p0/m, z1.d
; CHECK-NEXT:    lsl x9, x9, #3
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    st1d { z0.d }, p0, [sp]
; CHECK-NEXT:    cmp x8, #4
; CHECK-NEXT:    str q2, [x10, x9]
; CHECK-NEXT:    mov w9, #4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [sp]
; CHECK-NEXT:    csel x9, x8, x9, lo
; CHECK-NEXT:    lsl x9, x9, #3
; CHECK-NEXT:    addvl x10, sp, #1
; CHECK-NEXT:    cmp x8, #6
; CHECK-NEXT:    st1d { z0.d }, p0, [sp, #1, mul vl]
; CHECK-NEXT:    str q3, [x10, x9]
; CHECK-NEXT:    mov w9, #6
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [sp, #1, mul vl]
; CHECK-NEXT:    csel x8, x8, x9, lo
; CHECK-NEXT:    addvl x9, sp, #2
; CHECK-NEXT:    lsl x8, x8, #3
; CHECK-NEXT:    st1d { z0.d }, p0, [sp, #2, mul vl]
; CHECK-NEXT:    str q4, [x9, x8]
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [sp, #2, mul vl]
; CHECK-NEXT:    addvl sp, sp, #3
; CHECK-NEXT:   .cfi_def_cfa wsp, 16
; CHECK-NEXT:    ldr x29, [sp], #16 // 8-byte Folded Reload
; CHECK-NEXT:   .cfi_def_cfa_offset 0
; CHECK-NEXT:   .cfi_restore w29
; CHECK-NEXT:    ret




  %r = call <vscale x 2 x double> @llvm.experimental.vector.insert.nxv2f64.v8f64(<vscale x 2 x double> %a, <8 x double> %b, i64 0)
  ret <vscale x 2 x double> %r
}
