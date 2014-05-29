; RUN: opt < %s -loop-unroll -indvars -disable-output

@b = external global i32, align 4

; Function Attrs: nounwind uwtable
define void @fn1() #0 {
entry:
  br label %for.cond1thread-pre-split

for.cond1thread-pre-split:                        ; preds = %for.inc8, %entry
  %storemerge1 = phi i32 [ 0, %entry ], [ %inc9, %for.inc8 ]
  br i1 undef, label %for.inc8, label %for.cond2.preheader.lr.ph

for.cond2.preheader.lr.ph:                        ; preds = %for.cond1thread-pre-split
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.inc5, %for.cond2.preheader.lr.ph
  br label %for.cond2

for.cond2:                                        ; preds = %for.body3, %for.cond2.preheader
  %storemerge = phi i32 [ %add, %for.body3 ], [ 0, %for.cond2.preheader ]
  %cmp = icmp slt i32 %storemerge, 1
  br i1 %cmp, label %for.body3, label %for.inc5

for.body3:                                        ; preds = %for.cond2
  %tobool4 = icmp eq i32 %storemerge, 0
  %add = add nsw i32 %storemerge, 1
  br i1 %tobool4, label %for.cond2, label %if.then

if.then:                                          ; preds = %for.body3
  store i32 %storemerge1, i32* @b, align 4
  ret void

for.inc5:                                         ; preds = %for.cond2
  br i1 undef, label %for.cond1.for.inc8_crit_edge, label %for.cond2.preheader

for.cond1.for.inc8_crit_edge:                     ; preds = %for.inc5
  br label %for.inc8

for.inc8:                                         ; preds = %for.cond1.for.inc8_crit_edge, %for.cond1thread-pre-split
  %inc9 = add nsw i32 %storemerge1, 1
  br label %for.cond1thread-pre-split
}
