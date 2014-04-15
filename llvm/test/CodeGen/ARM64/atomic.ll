; RUN: llc < %s -march=arm64 -verify-machineinstrs -mcpu=cyclone | FileCheck %s

define i32 @val_compare_and_swap(i32* %p) {
; CHECK-LABEL: val_compare_and_swap:
; CHECK: orr    [[NEWVAL_REG:w[0-9]+]], wzr, #0x4
; CHECK: orr    [[OLDVAL_REG:w[0-9]+]], wzr, #0x7
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxr   [[RESULT:w[0-9]+]], [x0]
; CHECK: cmp    [[RESULT]], [[OLDVAL_REG]]
; CHECK: b.ne   [[LABEL2:.?LBB[0-9]+_[0-9]+]]
; CHECK: stxr   [[SCRATCH_REG:w[0-9]+]], [[NEWVAL_REG]], [x0]
; CHECK: cbnz   [[SCRATCH_REG]], [[LABEL]]
; CHECK: [[LABEL2]]:
  %val = cmpxchg i32* %p, i32 7, i32 4 acquire acquire
  ret i32 %val
}

define i64 @val_compare_and_swap_64(i64* %p) {
; CHECK-LABEL: val_compare_and_swap_64:
; CHECK: orr    [[NEWVAL_REG:x[0-9]+]], xzr, #0x4
; CHECK: orr    [[OLDVAL_REG:x[0-9]+]], xzr, #0x7
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxr   [[RESULT:x[0-9]+]], [x0]
; CHECK: cmp    [[RESULT]], [[OLDVAL_REG]]
; CHECK: b.ne   [[LABEL2:.?LBB[0-9]+_[0-9]+]]
; CHECK-NOT: stxr [[NEWVAL_REG]], [[NEWVAL_REG]]
; CHECK: stxr   [[SCRATCH_REG:w[0-9]+]], [[NEWVAL_REG]], [x0]
; CHECK: cbnz   [[SCRATCH_REG]], [[LABEL]]
; CHECK: [[LABEL2]]:
  %val = cmpxchg i64* %p, i64 7, i64 4 monotonic monotonic
  ret i64 %val
}

define i32 @fetch_and_nand(i32* %p) {
; CHECK-LABEL: fetch_and_nand:
; CHECK: orr    [[OLDVAL_REG:w[0-9]+]], wzr, #0x7
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxr   w[[DEST_REG:[0-9]+]], [x0]
; CHECK: bic    [[SCRATCH2_REG:w[0-9]+]], [[OLDVAL_REG]], w[[DEST_REG]]
; CHECK-NOT: stlxr [[SCRATCH2_REG]], [[SCRATCH2_REG]]
; CHECK: stlxr   [[SCRATCH_REG:w[0-9]+]], [[SCRATCH2_REG]], [x0]
; CHECK: cbnz   [[SCRATCH_REG]], [[LABEL]]
; CHECK: mov    x0, x[[DEST_REG]]
  %val = atomicrmw nand i32* %p, i32 7 release
  ret i32 %val
}

define i64 @fetch_and_nand_64(i64* %p) {
; CHECK-LABEL: fetch_and_nand_64:
; CHECK: orr    [[OLDVAL_REG:x[0-9]+]], xzr, #0x7
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxr   [[DEST_REG:x[0-9]+]], [x0]
; CHECK: bic    [[SCRATCH2_REG:x[0-9]+]], [[OLDVAL_REG]], [[DEST_REG]]
; CHECK: stlxr   [[SCRATCH_REG:w[0-9]+]], [[SCRATCH2_REG]], [x0]
; CHECK: cbnz   [[SCRATCH_REG]], [[LABEL]]
; CHECK: mov    x0, [[DEST_REG]]
  %val = atomicrmw nand i64* %p, i64 7 acq_rel
  ret i64 %val
}

define i32 @fetch_and_or(i32* %p) {
; CHECK-LABEL: fetch_and_or:
; CHECK: movz   [[OLDVAL_REG:w[0-9]+]], #5
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxr   w[[DEST_REG:[0-9]+]], [x0]
; CHECK: orr    [[SCRATCH2_REG:w[0-9]+]], w[[DEST_REG]], [[OLDVAL_REG]]
; CHECK-NOT: stlxr [[SCRATCH2_REG]], [[SCRATCH2_REG]]
; CHECK: stlxr [[SCRATCH_REG:w[0-9]+]], [[SCRATCH2_REG]], [x0]
; CHECK: cbnz   [[SCRATCH_REG]], [[LABEL]]
; CHECK: mov    x0, x[[DEST_REG]]
  %val = atomicrmw or i32* %p, i32 5 seq_cst
  ret i32 %val
}

define i64 @fetch_and_or_64(i64* %p) {
; CHECK: fetch_and_or_64:
; CHECK: orr    [[OLDVAL_REG:x[0-9]+]], xzr, #0x7
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxr   [[DEST_REG:x[0-9]+]], [x0]
; CHECK: orr    [[SCRATCH2_REG:x[0-9]+]], [[DEST_REG]], [[OLDVAL_REG]]
; CHECK: stxr   [[SCRATCH_REG:w[0-9]+]], [[SCRATCH2_REG]], [x0]
; CHECK: cbnz   [[SCRATCH_REG]], [[LABEL]]
; CHECK: mov    x0, [[DEST_REG]]
  %val = atomicrmw or i64* %p, i64 7 monotonic
  ret i64 %val
}

define void @acquire_fence() {
   fence acquire
   ret void
   ; CHECK-LABEL: acquire_fence:
   ; CHECK: dmb ishld
}

define void @release_fence() {
   fence release
   ret void
   ; CHECK-LABEL: release_fence:
   ; CHECK: dmb ish{{$}}
}

define void @seq_cst_fence() {
   fence seq_cst
   ret void
   ; CHECK-LABEL: seq_cst_fence:
   ; CHECK: dmb ish{{$}}
}

define i32 @atomic_load(i32* %p) {
   %r = load atomic i32* %p seq_cst, align 4
   ret i32 %r
   ; CHECK-LABEL: atomic_load:
   ; CHECK: ldar
}

define i8 @atomic_load_relaxed_8(i8* %p, i32 %off32) {
; CHECK-LABEL: atomic_load_relaxed_8:
  %ptr_unsigned = getelementptr i8* %p, i32 4095
  %val_unsigned = load atomic i8* %ptr_unsigned monotonic, align 1
; CHECK: ldrb {{w[0-9]+}}, [x0, #4095]

  %ptr_regoff = getelementptr i8* %p, i32 %off32
  %val_regoff = load atomic i8* %ptr_regoff unordered, align 1
  %tot1 = add i8 %val_unsigned, %val_regoff
  ; FIXME: syntax is incorrect: "sxtw" should not be able to go with an x-reg.
; CHECK: ldrb {{w[0-9]+}}, [x0, x1, sxtw]

  %ptr_unscaled = getelementptr i8* %p, i32 -256
  %val_unscaled = load atomic i8* %ptr_unscaled monotonic, align 1
  %tot2 = add i8 %tot1, %val_unscaled
; CHECK: ldurb {{w[0-9]+}}, [x0, #-256]

  %ptr_random = getelementptr i8* %p, i32 1191936 ; 0x123000 (i.e. ADD imm)
  %val_random = load atomic i8* %ptr_random unordered, align 1
  %tot3 = add i8 %tot2, %val_random
; CHECK: add x[[ADDR:[0-9]+]], x0, #1191936
; CHECK: ldrb {{w[0-9]+}}, [x[[ADDR]]]

  ret i8 %tot3
}

define i16 @atomic_load_relaxed_16(i16* %p, i32 %off32) {
; CHECK-LABEL: atomic_load_relaxed_16:
  %ptr_unsigned = getelementptr i16* %p, i32 4095
  %val_unsigned = load atomic i16* %ptr_unsigned monotonic, align 2
; CHECK: ldrh {{w[0-9]+}}, [x0, #8190]

  %ptr_regoff = getelementptr i16* %p, i32 %off32
  %val_regoff = load atomic i16* %ptr_regoff unordered, align 2
  %tot1 = add i16 %val_unsigned, %val_regoff
  ; FIXME: syntax is incorrect: "sxtw" should not be able to go with an x-reg.
; CHECK: ldrh {{w[0-9]+}}, [x0, x1, sxtw #1]

  %ptr_unscaled = getelementptr i16* %p, i32 -128
  %val_unscaled = load atomic i16* %ptr_unscaled monotonic, align 2
  %tot2 = add i16 %tot1, %val_unscaled
; CHECK: ldurh {{w[0-9]+}}, [x0, #-256]

  %ptr_random = getelementptr i16* %p, i32 595968 ; 0x123000/2 (i.e. ADD imm)
  %val_random = load atomic i16* %ptr_random unordered, align 2
  %tot3 = add i16 %tot2, %val_random
; CHECK: add x[[ADDR:[0-9]+]], x0, #1191936
; CHECK: ldrh {{w[0-9]+}}, [x[[ADDR]]]

  ret i16 %tot3
}

define i32 @atomic_load_relaxed_32(i32* %p, i32 %off32) {
; CHECK-LABEL: atomic_load_relaxed_32:
  %ptr_unsigned = getelementptr i32* %p, i32 4095
  %val_unsigned = load atomic i32* %ptr_unsigned monotonic, align 4
; CHECK: ldr {{w[0-9]+}}, [x0, #16380]

  %ptr_regoff = getelementptr i32* %p, i32 %off32
  %val_regoff = load atomic i32* %ptr_regoff unordered, align 4
  %tot1 = add i32 %val_unsigned, %val_regoff
  ; FIXME: syntax is incorrect: "sxtw" should not be able to go with an x-reg.
; CHECK: ldr {{w[0-9]+}}, [x0, x1, sxtw #2]

  %ptr_unscaled = getelementptr i32* %p, i32 -64
  %val_unscaled = load atomic i32* %ptr_unscaled monotonic, align 4
  %tot2 = add i32 %tot1, %val_unscaled
; CHECK: ldur {{w[0-9]+}}, [x0, #-256]

  %ptr_random = getelementptr i32* %p, i32 297984 ; 0x123000/4 (i.e. ADD imm)
  %val_random = load atomic i32* %ptr_random unordered, align 4
  %tot3 = add i32 %tot2, %val_random
; CHECK: add x[[ADDR:[0-9]+]], x0, #1191936
; CHECK: ldr {{w[0-9]+}}, [x[[ADDR]]]

  ret i32 %tot3
}

define i64 @atomic_load_relaxed_64(i64* %p, i32 %off32) {
; CHECK-LABEL: atomic_load_relaxed_64:
  %ptr_unsigned = getelementptr i64* %p, i32 4095
  %val_unsigned = load atomic i64* %ptr_unsigned monotonic, align 8
; CHECK: ldr {{x[0-9]+}}, [x0, #32760]

  %ptr_regoff = getelementptr i64* %p, i32 %off32
  %val_regoff = load atomic i64* %ptr_regoff unordered, align 8
  %tot1 = add i64 %val_unsigned, %val_regoff
  ; FIXME: syntax is incorrect: "sxtw" should not be able to go with an x-reg.
; CHECK: ldr {{x[0-9]+}}, [x0, x1, sxtw #3]

  %ptr_unscaled = getelementptr i64* %p, i32 -32
  %val_unscaled = load atomic i64* %ptr_unscaled monotonic, align 8
  %tot2 = add i64 %tot1, %val_unscaled
; CHECK: ldur {{x[0-9]+}}, [x0, #-256]

  %ptr_random = getelementptr i64* %p, i32 148992 ; 0x123000/8 (i.e. ADD imm)
  %val_random = load atomic i64* %ptr_random unordered, align 8
  %tot3 = add i64 %tot2, %val_random
; CHECK: add x[[ADDR:[0-9]+]], x0, #1191936
; CHECK: ldr {{x[0-9]+}}, [x[[ADDR]]]

  ret i64 %tot3
}


define void @atomc_store(i32* %p) {
   store atomic i32 4, i32* %p seq_cst, align 4
   ret void
   ; CHECK-LABEL: atomc_store:
   ; CHECK: stlr
}

define void @atomic_store_relaxed_8(i8* %p, i32 %off32, i8 %val) {
; CHECK-LABEL: atomic_store_relaxed_8:
  %ptr_unsigned = getelementptr i8* %p, i32 4095
  store atomic i8 %val, i8* %ptr_unsigned monotonic, align 1
; CHECK: strb {{w[0-9]+}}, [x0, #4095]

  %ptr_regoff = getelementptr i8* %p, i32 %off32
  store atomic i8 %val, i8* %ptr_regoff unordered, align 1
  ; FIXME: syntax is incorrect: "sxtw" should not be able to go with an x-reg.
; CHECK: strb {{w[0-9]+}}, [x0, x1, sxtw]

  %ptr_unscaled = getelementptr i8* %p, i32 -256
  store atomic i8 %val, i8* %ptr_unscaled monotonic, align 1
; CHECK: sturb {{w[0-9]+}}, [x0, #-256]

  %ptr_random = getelementptr i8* %p, i32 1191936 ; 0x123000 (i.e. ADD imm)
  store atomic i8 %val, i8* %ptr_random unordered, align 1
; CHECK: add x[[ADDR:[0-9]+]], x0, #1191936
; CHECK: strb {{w[0-9]+}}, [x[[ADDR]]]

  ret void
}

define void @atomic_store_relaxed_16(i16* %p, i32 %off32, i16 %val) {
; CHECK-LABEL: atomic_store_relaxed_16:
  %ptr_unsigned = getelementptr i16* %p, i32 4095
  store atomic i16 %val, i16* %ptr_unsigned monotonic, align 2
; CHECK: strh {{w[0-9]+}}, [x0, #8190]

  %ptr_regoff = getelementptr i16* %p, i32 %off32
  store atomic i16 %val, i16* %ptr_regoff unordered, align 2
  ; FIXME: syntax is incorrect: "sxtw" should not be able to go with an x-reg.
; CHECK: strh {{w[0-9]+}}, [x0, x1, sxtw #1]

  %ptr_unscaled = getelementptr i16* %p, i32 -128
  store atomic i16 %val, i16* %ptr_unscaled monotonic, align 2
; CHECK: sturh {{w[0-9]+}}, [x0, #-256]

  %ptr_random = getelementptr i16* %p, i32 595968 ; 0x123000/2 (i.e. ADD imm)
  store atomic i16 %val, i16* %ptr_random unordered, align 2
; CHECK: add x[[ADDR:[0-9]+]], x0, #1191936
; CHECK: strh {{w[0-9]+}}, [x[[ADDR]]]

  ret void
}

define void @atomic_store_relaxed_32(i32* %p, i32 %off32, i32 %val) {
; CHECK-LABEL: atomic_store_relaxed_32:
  %ptr_unsigned = getelementptr i32* %p, i32 4095
  store atomic i32 %val, i32* %ptr_unsigned monotonic, align 4
; CHECK: str {{w[0-9]+}}, [x0, #16380]

  %ptr_regoff = getelementptr i32* %p, i32 %off32
  store atomic i32 %val, i32* %ptr_regoff unordered, align 4
  ; FIXME: syntax is incorrect: "sxtw" should not be able to go with an x-reg.
; CHECK: str {{w[0-9]+}}, [x0, x1, sxtw #2]

  %ptr_unscaled = getelementptr i32* %p, i32 -64
  store atomic i32 %val, i32* %ptr_unscaled monotonic, align 4
; CHECK: stur {{w[0-9]+}}, [x0, #-256]

  %ptr_random = getelementptr i32* %p, i32 297984 ; 0x123000/4 (i.e. ADD imm)
  store atomic i32 %val, i32* %ptr_random unordered, align 4
; CHECK: add x[[ADDR:[0-9]+]], x0, #1191936
; CHECK: str {{w[0-9]+}}, [x[[ADDR]]]

  ret void
}

define void @atomic_store_relaxed_64(i64* %p, i32 %off32, i64 %val) {
; CHECK-LABEL: atomic_store_relaxed_64:
  %ptr_unsigned = getelementptr i64* %p, i32 4095
  store atomic i64 %val, i64* %ptr_unsigned monotonic, align 8
; CHECK: str {{x[0-9]+}}, [x0, #32760]

  %ptr_regoff = getelementptr i64* %p, i32 %off32
  store atomic i64 %val, i64* %ptr_regoff unordered, align 8
  ; FIXME: syntax is incorrect: "sxtw" should not be able to go with an x-reg.
; CHECK: str {{x[0-9]+}}, [x0, x1, sxtw #3]

  %ptr_unscaled = getelementptr i64* %p, i32 -32
  store atomic i64 %val, i64* %ptr_unscaled monotonic, align 8
; CHECK: stur {{x[0-9]+}}, [x0, #-256]

  %ptr_random = getelementptr i64* %p, i32 148992 ; 0x123000/8 (i.e. ADD imm)
  store atomic i64 %val, i64* %ptr_random unordered, align 8
; CHECK: add x[[ADDR:[0-9]+]], x0, #1191936
; CHECK: str {{x[0-9]+}}, [x[[ADDR]]]

  ret void
}

; rdar://11531169
; rdar://11531308

%"class.X::Atomic" = type { %struct.x_atomic_t }
%struct.x_atomic_t = type { i32 }

@counter = external hidden global %"class.X::Atomic", align 4

define i32 @next_id() nounwind optsize ssp align 2 {
entry:
  %0 = atomicrmw add i32* getelementptr inbounds (%"class.X::Atomic"* @counter, i64 0, i32 0, i32 0), i32 1 seq_cst
  %add.i = add i32 %0, 1
  %tobool = icmp eq i32 %add.i, 0
  br i1 %tobool, label %if.else, label %return

if.else:                                          ; preds = %entry
  %1 = atomicrmw add i32* getelementptr inbounds (%"class.X::Atomic"* @counter, i64 0, i32 0, i32 0), i32 1 seq_cst
  %add.i2 = add i32 %1, 1
  br label %return

return:                                           ; preds = %if.else, %entry
  %retval.0 = phi i32 [ %add.i2, %if.else ], [ %add.i, %entry ]
  ret i32 %retval.0
}
