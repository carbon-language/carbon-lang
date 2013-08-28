; Test the use of TEST UNDER MASK for 32-bit operations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

@g = global i32 0

; Check the lowest useful TMLL value.
define void @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: tmll %r2, 1
; CHECK: je {{\.L.*}}
; CHECK: br %r14
entry:
  %and = and i32 %a, 1
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the high end of the TMLL range.
define void @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: tmll %r2, 65535
; CHECK: jne {{\.L.*}}
; CHECK: br %r14
entry:
  %and = and i32 %a, 65535
  %cmp = icmp ne i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the lowest useful TMLH value, which is the next value up.
define void @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: tmlh %r2, 1
; CHECK: jne {{\.L.*}}
; CHECK: br %r14
entry:
  %and = and i32 %a, 65536
  %cmp = icmp ne i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the next value up again, which cannot use TM.
define void @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK-NOT: {{tm[lh].}}
; CHECK: br %r14
entry:
  %and = and i32 %a, 4294901759
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the high end of the TMLH range.
define void @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: tmlh %r2, 65535
; CHECK: je {{\.L.*}}
; CHECK: br %r14
entry:
  %and = and i32 %a, 4294901760
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}
