; RUN: opt -S %s -value-tracking-dom-conditions -licm -load-combine | FileCheck %s
; In pr24866.ll, we saw a crash when accessing a nullptr returned when
; asking for a dominator tree Node.  This reproducer is really fragile,
; but it's currently the best we have.

%struct.c_derived_tbl.2.5.8.11.14.17.23.38.59.80.92.98.104.107.155.183 = type { [256 x i32], [256 x i8] }


; Function Attrs: nounwind uwtable
define void @encode_one_blockX2(%struct.c_derived_tbl.2.5.8.11.14.17.23.38.59.80.92.98.104.107.155.183* nocapture readonly %actbl) #0 {
; CHECK-LABEL: @encode_one_blockX2
entry:
  br i1 false, label %L_KLOOP_01, label %L_KLOOP.preheader

L_KLOOP_01:                                       ; preds = %while.end, %entry
  br label %L_KLOOP.preheader

L_KLOOP_08:                                       ; preds = %while.end
  br label %L_KLOOP.preheader

L_KLOOP.preheader:                                ; preds = %L_KLOOP_08, %L_KLOOP_01, %entry
  %r.2.ph = phi i32 [ undef, %L_KLOOP_08 ], [ 0, %entry ], [ undef, %L_KLOOP_01 ]
  br label %L_KLOOP

L_KLOOP:                                          ; preds = %while.end, %L_KLOOP.preheader
  %r.2 = phi i32 [ 0, %while.end ], [ %r.2.ph, %L_KLOOP.preheader ]
  br i1 true, label %while.body, label %while.end

while.body:                                       ; preds = %while.body, %L_KLOOP
  br label %while.body

while.end:                                        ; preds = %L_KLOOP
  %shl105 = shl i32 %r.2, 4
  %add106 = add nsw i32 %shl105, undef
  %idxprom107 = sext i32 %add106 to i64
  %arrayidx108 = getelementptr inbounds %struct.c_derived_tbl.2.5.8.11.14.17.23.38.59.80.92.98.104.107.155.183, %struct.c_derived_tbl.2.5.8.11.14.17.23.38.59.80.92.98.104.107.155.183* %actbl, i64 0, i32 0, i64 %idxprom107
  %0 = load i32, i32* %arrayidx108, align 4
  %arrayidx110 = getelementptr inbounds %struct.c_derived_tbl.2.5.8.11.14.17.23.38.59.80.92.98.104.107.155.183, %struct.c_derived_tbl.2.5.8.11.14.17.23.38.59.80.92.98.104.107.155.183* %actbl, i64 0, i32 1, i64 %idxprom107
  %1 = load i8, i8* %arrayidx110, align 1
  indirectbr i8* undef, [label %L_KLOOP_DONE, label %L_KLOOP_01, label %L_KLOOP_08, label %L_KLOOP]

L_KLOOP_DONE:                                     ; preds = %while.end
  ret void
}
