; RUN: opt -simplifycfg %s -S -o - | FileCheck %s

define void @patatino() {

; CHECK-LABEL: @patatino
; CHECK:   br i1 undef, label %bb4, label %bb9.critedge
; CHECK: bb4:                                              ; preds = %bb4, %bb1
; CHECK-NEXT:   %.pr = phi i1 [ undef, %bb4 ], [ true, %bb1 ]
; CHECK-NEXT:   br i1 %.pr, label %bb4, label %bb6
; CHECK: bb6:                                              ; preds = %bb4
; CHECK-NEXT:   %tmp = or i64 undef, 1
; CHECK-NEXT:   %tmp8 = icmp ult i64 %tmp, 0
; CHECK-NEXT:   %or.cond = and i1 undef, %tmp8
; CHECK-NEXT:   %tmp13 = icmp ult i64 0, 0
; CHECK-NEXT:   %or.cond2 = and i1 %or.cond, %tmp13
; CHECK-NEXT:   %tmp15 = icmp ult i64 undef, 0
; CHECK-NEXT:   %or.cond3 = and i1 %or.cond2, %tmp15
; CHECK-NEXT:   %tmp19 = or i64 undef, 5
; CHECK-NEXT:   %tmp20 = icmp ult i64 %tmp19, 0
; CHECK-NEXT:   %or.cond4 = and i1 %or.cond3, %tmp20
; CHECK-NEXT:   %tmp22 = or i64 undef, 6
; CHECK-NEXT:   %tmp23 = icmp ult i64 %tmp22, 0
; CHECK-NEXT:   %or.cond5 = and i1 %or.cond4, %tmp23
; CHECK-NEXT:   %tmp25 = or i64 undef, 7
; CHECK-NEXT:   %tmp26 = icmp ult i64 %tmp25, 0
; CHECK-NEXT:   %or.cond6 = and i1 %or.cond5, %tmp26
; CHECK-NEXT:   %tmp28 = or i64 undef, 8
; CHECK-NEXT:   %tmp29 = icmp ult i64 %tmp28, 0
; CHECK-NEXT:   %or.cond7 = and i1 %or.cond6, %tmp29
; CHECK-NEXT:   %or.cond7.not = xor i1 %or.cond7, true
; CHECK-NEXT:   %.not = xor i1 undef, true
; CHECK-NEXT:   %brmerge = or i1 %or.cond7.not, %.not
; CHECK-NEXT:   %.not8 = xor i1 undef, true
; CHECK-NEXT:   %brmerge9 = or i1 %brmerge, %.not8
; CHECK-NEXT:   %.not10 = xor i1 undef, true
; CHECK-NEXT:   %brmerge11 = or i1 %brmerge9, %.not10
; CHECK-NEXT:   %.not12 = xor i1 undef, true
; CHECK-NEXT:   %brmerge13 = or i1 %brmerge11, %.not12
; CHECK-NEXT:   %.not14 = xor i1 undef, true
; CHECK-NEXT:   %brmerge15 = or i1 %brmerge13, %.not14
; CHECK-NEXT:   %.not16 = xor i1 undef, true
; CHECK-NEXT:   %brmerge17 = or i1 %brmerge15, %.not16
; CHECK-NEXT:   %.not18 = xor i1 undef, true
; CHECK-NEXT:   %brmerge19 = or i1 %brmerge17, %.not18
; CHECK-NEXT:   br i1 %brmerge19, label %bb11, label %bb1
; CHECK: bb9.critedge:                                     ; preds = %bb1
; CHECK-NEXT:   br label %bb11
; CHECK: bb11:                                             ; preds = %bb6, %bb9.critedge
; CHECK-NEXT:  ret void

  br label %bb1
bb1:                                              ; preds = %bb36, %0
  br label %bb2
bb2:                                              ; preds = %bb3, %bb1
  br i1 undef, label %bb4, label %bb3
bb3:                                              ; preds = %bb4, %bb2
  br i1 undef, label %bb2, label %bb5
bb4:                                              ; preds = %bb2
  switch i32 undef, label %bb3 [
  ]
bb5:                                              ; preds = %bb3
  br label %bb6
bb6:                                              ; preds = %bb5
  br i1 undef, label %bb7, label %bb9
bb7:                                              ; preds = %bb6
  %tmp = or i64 undef, 1
  %tmp8 = icmp ult i64 %tmp, 0
  br i1 %tmp8, label %bb12, label %bb9
bb9:                                              ; preds = %bb35, %bb34, %bb33, %bb32, %bb31, %bb30, %bb27, %bb24, %bb21, %bb18, %bb16, %bb14, %bb12, %bb7, %bb6
  br label %bb11
bb10:                                             ; preds = %bb36
  br label %bb11
bb11:                                             ; preds = %bb10, %bb9
  ret void
bb12:                                             ; preds = %bb7
  %tmp13 = icmp ult i64 0, 0
  br i1 %tmp13, label %bb14, label %bb9
bb14:                                             ; preds = %bb12
  %tmp15 = icmp ult i64 undef, 0
  br i1 %tmp15, label %bb16, label %bb9
bb16:                                             ; preds = %bb14
  %tmp17 = icmp ult i64 undef, 0
  br i1 %tmp17, label %bb18, label %bb9
bb18:                                             ; preds = %bb16
  %tmp19 = or i64 undef, 5
  %tmp20 = icmp ult i64 %tmp19, 0
  br i1 %tmp20, label %bb21, label %bb9
bb21:                                             ; preds = %bb18
  %tmp22 = or i64 undef, 6
  %tmp23 = icmp ult i64 %tmp22, 0
  br i1 %tmp23, label %bb24, label %bb9
bb24:                                             ; preds = %bb21
  %tmp25 = or i64 undef, 7
  %tmp26 = icmp ult i64 %tmp25, 0
  br i1 %tmp26, label %bb27, label %bb9
bb27:                                             ; preds = %bb24
  %tmp28 = or i64 undef, 8
  %tmp29 = icmp ult i64 %tmp28, 0
  br i1 %tmp29, label %bb30, label %bb9
bb30:                                             ; preds = %bb27
  br i1 undef, label %bb31, label %bb9
bb31:                                             ; preds = %bb30
  br i1 undef, label %bb32, label %bb9
bb32:                                             ; preds = %bb31
  br i1 undef, label %bb33, label %bb9
bb33:                                             ; preds = %bb32
  br i1 undef, label %bb34, label %bb9
bb34:                                             ; preds = %bb33
  br i1 undef, label %bb35, label %bb9
bb35:                                             ; preds = %bb34
  br i1 undef, label %bb36, label %bb9
bb36:                                             ; preds = %bb35
  br i1 undef, label %bb1, label %bb10
}
