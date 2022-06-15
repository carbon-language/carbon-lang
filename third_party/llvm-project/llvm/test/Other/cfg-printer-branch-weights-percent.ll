;RUN: opt < %s -dot-cfg -cfg-weights -cfg-dot-filename-prefix=%t 2>/dev/null > /dev/null
;RUN: FileCheck %s -input-file=%t.f.dot

define void @f(i32) {
entry:
  %check = icmp sgt i32 %0, 0
  br i1 %check, label %if, label %exit, !prof !0

; CHECK: label="0.50%"
; CHECK-NOT: ["];
if:                     ; preds = %entry
  br label %exit
; CHECK: label="99.50%"
; CHECK-NOT: ["];
exit:                   ; preds = %entry, %if
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 200}
