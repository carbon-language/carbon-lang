; RUN: llc < %s -mtriple=arm64-apple-iOS5.0

; CPSR is not allocatable so fast allocatable wouldn't mark them killed.
; rdar://9313272

define hidden void @t() nounwind {
entry:
  %cmp = icmp eq i32* null, undef
  %frombool = zext i1 %cmp to i8
  store i8 %frombool, i8* undef, align 1
  %tmp4 = load i8, i8* undef, align 1
  %tobool = trunc i8 %tmp4 to i1
  br i1 %tobool, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  br i1 undef, label %land.lhs.true14, label %if.end33

land.lhs.true14:                                  ; preds = %if.end
  unreachable

if.end33:                                         ; preds = %if.end
  unreachable
}
