; RUN: opt -jump-threading -S < %s | FileCheck %s
; RUN: opt -passes=jump-threading -S < %s | FileCheck %s

; Check the unreachable loop won't cause infinite loop
; in jump-threading when it tries to update the predecessors'
; profile metadata from a phi node.

define void @unreachable_single_bb_loop() {
; CHECK-LABEL: @unreachable_single_bb_loop(
bb:
  %tmp = call i32 @a()
  %tmp1 = icmp eq i32 %tmp, 1
  br i1 %tmp1, label %bb5, label %bb8

; unreachable single bb loop.
bb2:                                              ; preds = %bb2
  %tmp4 = icmp ne i32 %tmp, 1
  switch i1 %tmp4, label %bb2 [
  i1 0, label %bb5
  i1 1, label %bb8
  ]

bb5:                                              ; preds = %bb2, %bb
  %tmp6 = phi i1 [ %tmp1, %bb ], [ false, %bb2 ]
  br i1 %tmp6, label %bb8, label %bb7, !prof !0

bb7:                                              ; preds = %bb5
  br label %bb8

bb8:                                              ; preds = %bb8, %bb7, %bb5, %bb2
  ret void
}

define void @unreachable_multi_bbs_loop() {
; CHECK-LABEL: @unreachable_multi_bbs_loop(
bb:
  %tmp = call i32 @a()
  %tmp1 = icmp eq i32 %tmp, 1
  br i1 %tmp1, label %bb5, label %bb8

; unreachable two bbs loop.
bb3:                                              ; preds = %bb2
  br label %bb2

bb2:                                              ; preds = %bb3
  %tmp4 = icmp ne i32 %tmp, 1
  switch i1 %tmp4, label %bb3 [
  i1 0, label %bb5
  i1 1, label %bb8
  ]

bb5:                                              ; preds = %bb2, %bb
  %tmp6 = phi i1 [ %tmp1, %bb ], [ false, %bb2 ]
  br i1 %tmp6, label %bb8, label %bb7, !prof !0

bb7:                                              ; preds = %bb5
  br label %bb8

bb8:                                              ; preds = %bb8, %bb7, %bb5, %bb2
  ret void
}
declare i32 @a()

; This gets into a state that could cause instruction simplify
; to hang - an insertelement instruction has itself as an operand.

define void @PR48362() {
; CHECK-LABEL: @PR48362(
cleanup1491:                                      ; preds = %for.body1140
  switch i32 0, label %cleanup2343.loopexit4 [
  i32 0, label %cleanup.cont1500
  i32 128, label %lbl_555.loopexit
  ]

cleanup.cont1500:                                 ; preds = %cleanup1491
  unreachable

lbl_555.loopexit:                                 ; preds = %cleanup1491
  br label %for.body1509

for.body1509:                                     ; preds = %for.inc2340, %lbl_555.loopexit
  %l_580.sroa.0.0 = phi <4 x i32> [ <i32 1684658741, i32 1684658741, i32 1684658741, i32 1684658741>, %lbl_555.loopexit ], [ %l_580.sroa.0.2, %for.inc2340 ]
  %p_55.addr.10 = phi i16 [ 0, %lbl_555.loopexit ], [ %p_55.addr.11, %for.inc2340 ]
  %i82 = load i32, i32* undef, align 1
  %tobool1731.not = icmp eq i32 %i82, 0
  br i1 %tobool1731.not, label %if.end1733, label %if.then1732

if.then1732:                                      ; preds = %for.body1509
  br label %cleanup2329

if.end1733:                                       ; preds = %for.body1509
  %tobool1735.not = icmp eq i16 %p_55.addr.10, 0
  br i1 %tobool1735.not, label %if.then1736, label %if.else1904

if.then1736:                                      ; preds = %if.end1733
  br label %cleanup2329

if.else1904:                                      ; preds = %if.end1733
  br label %for.body1911

for.body1911:                                     ; preds = %if.else1904
  %l_580.sroa.0.4.vec.extract683 = extractelement <4 x i32> %l_580.sroa.0.0, i32 2
  %xor2107 = xor i32 undef, %l_580.sroa.0.4.vec.extract683
  br label %land.end2173

land.end2173:                                     ; preds = %for.body1911
  br i1 undef, label %if.end2178, label %cleanup2297

if.end2178:                                       ; preds = %land.end2173
  %l_580.sroa.0.2.vec.insert = insertelement <4 x i32> %l_580.sroa.0.0, i32 undef, i32 1
  br label %cleanup2297

cleanup2297:                                      ; preds = %if.end2178, %land.end2173
  %l_580.sroa.0.1 = phi <4 x i32> [ %l_580.sroa.0.2.vec.insert, %if.end2178 ], [ %l_580.sroa.0.0, %land.end2173 ]
  br label %cleanup2329

cleanup2329:                                      ; preds = %cleanup2297, %if.then1736, %if.then1732
  %l_580.sroa.0.2 = phi <4 x i32> [ %l_580.sroa.0.0, %if.then1736 ], [ %l_580.sroa.0.1, %cleanup2297 ], [ %l_580.sroa.0.0, %if.then1732 ]
  %cleanup.dest.slot.11 = phi i32 [ 0, %if.then1736 ], [ undef, %cleanup2297 ], [ 129, %if.then1732 ]
  %p_55.addr.11 = phi i16 [ %p_55.addr.10, %if.then1736 ], [ undef, %cleanup2297 ], [ %p_55.addr.10, %if.then1732 ]
  switch i32 %cleanup.dest.slot.11, label %cleanup2343.loopexit [
  i32 0, label %cleanup.cont2339
  i32 129, label %crit_edge114
  ]

cleanup.cont2339:                                 ; preds = %cleanup2329
  br label %for.inc2340

for.inc2340:                                      ; preds = %cleanup.cont2339
  br i1 undef, label %for.body1509, label %crit_edge115

crit_edge114:                                     ; preds = %cleanup2329
  unreachable

crit_edge115:                                     ; preds = %for.inc2340
  unreachable

cleanup2343.loopexit:                             ; preds = %cleanup2329
  unreachable

cleanup2343.loopexit4:                            ; preds = %cleanup1491
  unreachable
}

!0 = !{!"branch_weights", i32 2146410443, i32 1073205}
