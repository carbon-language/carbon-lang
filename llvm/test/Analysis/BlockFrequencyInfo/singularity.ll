; RUN: opt < %s -analyze -block-freq | FileCheck %s
; PR16402

define void @test1(i32 %n) nounwind {
entry:
  %call = tail call i32* @cond() nounwind
  %tobool = icmp eq i32* %call, null
  br i1 %tobool, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %call1 = tail call i32* @cond() nounwind
  %tobool2 = icmp eq i32* %call1, null
  br i1 %tobool2, label %land.lhs.true3, label %if.end

land.lhs.true3:                                   ; preds = %land.lhs.true
  %call4 = tail call i32* @cond() nounwind
  %tobool5 = icmp eq i32* %call4, null
  br i1 %tobool5, label %land.lhs.true6, label %if.end

land.lhs.true6:                                   ; preds = %land.lhs.true3
  %call7 = tail call i32* @cond() nounwind
  %tobool8 = icmp eq i32* %call7, null
  br i1 %tobool8, label %land.lhs.true9, label %if.end

land.lhs.true9:                                   ; preds = %land.lhs.true6
  %call10 = tail call i32* @cond() nounwind
  %tobool11 = icmp eq i32* %call10, null
  br i1 %tobool11, label %land.lhs.true12, label %if.end

land.lhs.true12:                                  ; preds = %land.lhs.true9
  %call13 = tail call i32* @cond() nounwind
  %tobool14 = icmp eq i32* %call13, null
  br i1 %tobool14, label %land.lhs.true15, label %if.end

land.lhs.true15:                                  ; preds = %land.lhs.true12
  %call16 = tail call i32* @cond() nounwind
  %tobool17 = icmp eq i32* %call16, null
  br i1 %tobool17, label %for.cond.preheader, label %if.end

for.cond.preheader:                               ; preds = %land.lhs.true15
  %cmp21 = icmp eq i32 %n, 0
  br i1 %cmp21, label %for.end, label %for.body

for.body:                                         ; preds = %for.cond.preheader, %for.body
  %i.022 = phi i32 [ %inc, %for.body ], [ 0, %for.cond.preheader ]
  %call18 = tail call i32 @call() nounwind
  %inc = add nsw i32 %i.022, 1
  %cmp = icmp eq i32 %inc, %n
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %for.cond.preheader
  %call19 = tail call i32* @cond() nounwind
  br label %if.end

if.end:                                           ; preds = %land.lhs.true15, %land.lhs.true12, %land.lhs.true9, %land.lhs.true6, %land.lhs.true3, %land.lhs.true, %entry, %for.end
  ret void

; CHECK: entry = 1024
; CHECK-NOT: for.body = 0
; CHECK-NOT: for.end = 0
}

declare i32* @cond() nounwind

declare i32 @call() nounwind
