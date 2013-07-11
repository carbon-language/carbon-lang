;RUN: opt -S %s -indvars | FileCheck %s

; Function Attrs: nounwind uwtable
define void @foo() #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i16 [ 0, %entry ], [ %inc, %for.body ]
  %conv2 = sext i16 %i.01 to i32
  call void @bar(i32 %conv2) #1
  %inc = add i16 %i.01, 1
;CHECK-NOT: %lftr.wideiv = trunc i32 %indvars.iv.next to i16
;CHECK: %exitcond = icmp ne i32 %indvars.iv.next, 512
  %cmp = icmp slt i16 %inc, 512
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

declare void @bar(i32)

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }
