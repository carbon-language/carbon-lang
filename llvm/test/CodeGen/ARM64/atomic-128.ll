; RUN: llc < %s -march=arm64 -mtriple=arm64-linux-gnu -verify-machineinstrs | FileCheck %s

@var = global i128 0

define i128 @val_compare_and_swap(i128* %p, i128 %oldval, i128 %newval) {
; CHECK-LABEL: val_compare_and_swap:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp   [[RESULTLO:x[0-9]+]], [[RESULTHI:x[0-9]+]], [x0]
; CHECK: cmp    [[RESULTLO]], x2
; CHECK: sbc    xzr, [[RESULTHI]], x3
; CHECK: b.ne   [[LABEL2:.?LBB[0-9]+_[0-9]+]]
; CHECK: stxp   [[SCRATCH_RES:w[0-9]+]], x4, x5, [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]
; CHECK: [[LABEL2]]:
  %val = cmpxchg i128* %p, i128 %oldval, i128 %newval acquire acquire
  ret i128 %val
}

define void @fetch_and_nand(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_nand:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: bic    [[SCRATCH_REGLO:x[0-9]+]], x2, [[DEST_REGLO]]
; CHECK: bic    [[SCRATCH_REGHI:x[0-9]+]], x3, [[DEST_REGHI]]
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK: str    [[DEST_REGHI]]
; CHECK: str    [[DEST_REGLO]]
  %val = atomicrmw nand i128* %p, i128 %bits release
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_or(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_or:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: orr    [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2
; CHECK: orr    [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK: str    [[DEST_REGHI]]
; CHECK: str    [[DEST_REGLO]]
  %val = atomicrmw or i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_add(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_add:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: adds   [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2
; CHECK: adc    [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK: str    [[DEST_REGHI]]
; CHECK: str    [[DEST_REGLO]]
  %val = atomicrmw add i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_sub(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_sub:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: subs   [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2
; CHECK: sbc    [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK: str    [[DEST_REGHI]]
; CHECK: str    [[DEST_REGLO]]
  %val = atomicrmw sub i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_min(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_min:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: cmp    [[DEST_REGLO]], x2
; CHECK: sbc    xzr, [[DEST_REGHI]], x3
; CHECK: csel   [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2, lt
; CHECK: csel   [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3, lt
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK: str    [[DEST_REGHI]]
; CHECK: str    [[DEST_REGLO]]
  %val = atomicrmw min i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_max(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_max:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: cmp    [[DEST_REGLO]], x2
; CHECK: sbc    xzr, [[DEST_REGHI]], x3
; CHECK: csel   [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2, gt
; CHECK: csel   [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3, gt
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK: str    [[DEST_REGHI]]
; CHECK: str    [[DEST_REGLO]]
  %val = atomicrmw max i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_umin(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_umin:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: cmp    [[DEST_REGLO]], x2
; CHECK: sbc    xzr, [[DEST_REGHI]], x3
; CHECK: csel   [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2, cc
; CHECK: csel   [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3, cc
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK: str    [[DEST_REGHI]]
; CHECK: str    [[DEST_REGLO]]
  %val = atomicrmw umin i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_umax(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_umax:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: cmp    [[DEST_REGLO]], x2
; CHECK: sbc    xzr, [[DEST_REGHI]], x3
; CHECK: csel   [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2, hi
; CHECK: csel   [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3, hi
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK: str    [[DEST_REGHI]]
; CHECK: str    [[DEST_REGLO]]
  %val = atomicrmw umax i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define i128 @atomic_load_seq_cst(i128* %p) {
; CHECK-LABEL: atomic_load_seq_cst:
; CHECK-NOT: dmb
; CHECK-LABEL: ldaxp
; CHECK-NOT: dmb
   %r = load atomic i128* %p seq_cst, align 16
   ret i128 %r
}

define i128 @atomic_load_relaxed(i128* %p) {
; CHECK-LABEL: atomic_load_relaxed:
; CHECK-NOT: dmb
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0]
; CHECK: orr [[SAMELO:x[0-9]+]], [[LO]], xzr
; CHECK: orr [[SAMEHI:x[0-9]+]], [[HI]], xzr
; CHECK: stxp [[SUCCESS:w[0-9]+]], [[SAMELO]], [[SAMEHI]], [x0]
; CHECK: cbnz [[SUCCESS]], [[LABEL]]
; CHECK-NOT: dmb
   %r = load atomic i128* %p monotonic, align 16
   ret i128 %r
}


define void @atomic_store_seq_cst(i128 %in, i128* %p) {
; CHECK-LABEL: atomic_store_seq_cst:
; CHECK-NOT: dmb
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp xzr, xzr, [x2]
; CHECK: stlxp [[SUCCESS:w[0-9]+]], x0, x1, [x2]
; CHECK: cbnz [[SUCCESS]], [[LABEL]]
; CHECK-NOT: dmb
   store atomic i128 %in, i128* %p seq_cst, align 16
   ret void
}

define void @atomic_store_release(i128 %in, i128* %p) {
; CHECK-LABEL: atomic_store_release:
; CHECK-NOT: dmb
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxp xzr, xzr, [x2]
; CHECK: stlxp [[SUCCESS:w[0-9]+]], x0, x1, [x2]
; CHECK: cbnz [[SUCCESS]], [[LABEL]]
; CHECK-NOT: dmb
   store atomic i128 %in, i128* %p release, align 16
   ret void
}

define void @atomic_store_relaxed(i128 %in, i128* %p) {
; CHECK-LABEL: atomic_store_relaxed:
; CHECK-NOT: dmb
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxp xzr, xzr, [x2]
; CHECK: stxp [[SUCCESS:w[0-9]+]], x0, x1, [x2]
; CHECK: cbnz [[SUCCESS]], [[LABEL]]
; CHECK-NOT: dmb
   store atomic i128 %in, i128* %p unordered, align 16
   ret void
}
