; RUN: llc -march=x86 -align-all-blocks=16 < %s | FileCheck %s

;CHECK: foo
;CHECK: .align  16, 0x90
;CHECK: .align  16, 0x90
;CHECK: .align  16, 0x90
;CHECK: ret
define i32 @foo(i32 %t, i32 %l) nounwind readnone ssp uwtable {
  %1 = icmp eq i32 %t, 0
  br i1 %1, label %4, label %2

; <label>:2                                       ; preds = %0
  %3 = add nsw i32 %t, 2
  ret i32 %3

; <label>:4                                       ; preds = %0
  %5 = icmp eq i32 %l, 0
  %. = select i1 %5, i32 0, i32 5
  ret i32 %.
}


