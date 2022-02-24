; RUN: opt < %s -newgvn
; PR12858

define void @fn5(i16 signext %p1, i8 signext %p2) nounwind uwtable {
entry:
  br i1 undef, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  br label %if.end

if.else:                                          ; preds = %entry
  %conv = sext i16 %p1 to i32
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %conv1 = sext i16 %p1 to i32
  br i1 undef, label %if.then3, label %if.else4

if.then3:                                         ; preds = %if.end
  br label %if.end12

if.else4:                                         ; preds = %if.end
  %conv7 = sext i8 %p2 to i32
  %cmp8 = icmp eq i32 %conv1, %conv7
  br i1 %cmp8, label %if.then10, label %if.end12

if.then10:                                        ; preds = %if.else4
  br label %if.end12

if.end12:                                         ; preds = %if.then10, %if.else4, %if.then3
  %conv13 = sext i8 %p2 to i32
  ret void
}
