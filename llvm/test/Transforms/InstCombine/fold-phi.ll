; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: no_crash
define float @no_crash(float %a) nounwind {
entry:
  br label %for.body

for.body:
  %sum.057 = phi float [ 0.000000e+00, %entry ], [ %add5, %bb0 ]
  %add5 = fadd float %sum.057, %a    ; PR14592
  br i1 undef, label %bb0, label %end

bb0:
  br label %for.body

end:
  ret float %add5
}

; CHECK-LABEL: @pr21377(
define void @pr21377(i32, i32) {
entry:
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.end.i, %entry
  %g.0.i = phi i64 [ 0, %entry ], [ %phitmp5.i, %while.end.i ]
  br i1 undef, label %fn2.exit, label %while.body.i

while.body.i:                                     ; preds = %while.cond.i
  %conv.i = zext i32 %0 to i64
  %phitmp3.i = or i64 %g.0.i, %conv.i
  br label %while.cond3.i

while.cond3.i:                                    ; preds = %while.cond3.i, %while.body.i
  %g.1.i = phi i64 [ %phitmp3.i, %while.body.i ], [ 0, %while.cond3.i ]
  br i1 undef, label %while.end.i, label %while.cond3.i

while.end.i:                                      ; preds = %while.cond3.i
  %conv.i.i = zext i32 %1 to i64
  %or7.i = or i64 %g.1.i, %conv.i.i
  %phitmp5.i = and i64 %or7.i, 4294967295
  br label %while.cond.i

fn2.exit:                                         ; preds = %while.cond.i
  ret void
}
