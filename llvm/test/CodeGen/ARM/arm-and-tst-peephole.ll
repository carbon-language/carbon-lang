; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6-apple-darwin | FileCheck -check-prefix=T2 %s

%struct.Foo = type { i8* }

define %struct.Foo* @_ZN3Foo7collectEj(%struct.Foo* %this, i32 %acc) nounwind readonly align 2 {
entry:
  %scevgep = getelementptr %struct.Foo* %this, i32 1
  br label %tailrecurse

tailrecurse:                                      ; preds = %sw.bb, %entry
  %lsr.iv2 = phi %struct.Foo* [ %scevgep3, %sw.bb ], [ %scevgep, %entry ]
  %lsr.iv = phi i32 [ %lsr.iv.next, %sw.bb ], [ 1, %entry ]
  %acc.tr = phi i32 [ %or, %sw.bb ], [ %acc, %entry ]
  %lsr.iv24 = bitcast %struct.Foo* %lsr.iv2 to i8**
  %scevgep5 = getelementptr i8** %lsr.iv24, i32 -1
  %tmp2 = load i8** %scevgep5
  %0 = ptrtoint i8* %tmp2 to i32

; CHECK:      ands r12, r12, #3
; CHECK-NEXT: beq LBB0_2

; T2:      movs r5, #3
; T2-NEXT: mov r6, r4
; T2-NEXT: ands r6, r5
; T2-NEXT: tst r4, r5
; T2-NEXT: beq LBB0_3

  %and = and i32 %0, 3
  %tst = icmp eq i32 %and, 0
  br i1 %tst, label %sw.bb, label %tailrecurse.switch

tailrecurse.switch:                               ; preds = %tailrecurse
  switch i32 %and, label %sw.epilog [
    i32 1, label %sw.bb
    i32 3, label %sw.bb6
    i32 2, label %sw.bb8
  ]

sw.bb:                                            ; preds = %tailrecurse.switch, %tailrecurse
  %shl = shl i32 %acc.tr, 1
  %or = or i32 %and, %shl
  %lsr.iv.next = add i32 %lsr.iv, 1
  %scevgep3 = getelementptr %struct.Foo* %lsr.iv2, i32 1
  br label %tailrecurse

sw.bb6:                                           ; preds = %tailrecurse.switch
  ret %struct.Foo* %lsr.iv2

sw.bb8:                                           ; preds = %tailrecurse.switch
  %tmp1 = add i32 %acc.tr, %lsr.iv
  %add.ptr11 = getelementptr inbounds %struct.Foo* %this, i32 %tmp1
  ret %struct.Foo* %add.ptr11

sw.epilog:                                        ; preds = %tailrecurse.switch
  ret %struct.Foo* undef
}
