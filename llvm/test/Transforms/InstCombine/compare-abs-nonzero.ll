; RUN: opt < %s -instcombine -S | grep -v {select}

define zeroext i1 @cmpabs(i64 %val) nounwind uwtable readnone ssp {
entry:
  %sub = sub nsw i64 0, %val
  %cmp = icmp slt i64 %val, 0
  %sub.val = select i1 %cmp, i64 %sub, i64 %val
  %tobool = icmp ne i64 %sub.val, 0
  ret i1 %tobool
}
