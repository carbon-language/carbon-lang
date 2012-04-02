; RUN: opt -loop-unswitch -disable-output -stats -info-output-file - < %s | FileCheck --check-prefix=STATS %s
; RUN: opt -S -loop-unswitch -verify-loop-info -verify-dom-info %s | FileCheck %s

; STATS: 1 loop-unswitch - Total number of instructions analyzed

; CHECK:       %0 = icmp eq i64 undef, 0
; CHECK-NEXT:  br i1 %0, label %"5", label %"4"

; CHECK:       "5":                                              ; preds = %entry
; CHECK-NEXT:  br label %"5.split"

; CHECK:       "5.split":                                        ; preds = %"5"
; CHECK-NEXT:  br label %"16"

; CHECK:       "16":                                             ; preds = %"22", %"5.split"
; CHECK-NEXT:  indirectbr i8* undef, [label %"22", label %"33"]

; CHECK:       "22":                                             ; preds = %"16"
; CHECK-NEXT:  br i1 %0, label %"16", label %"26"

; CHECK:       "26":                                             ; preds = %"22"
; CHECK-NEXT:  unreachable

define void @foo() {
entry:
  %0 = icmp eq i64 undef, 0
  br i1 %0, label %"5", label %"4"

"4":                                              ; preds = %entry
  unreachable

"5":                                              ; preds = %entry
  br label %"16"

"16":                                             ; preds = %"22", %"5"
  indirectbr i8* undef, [label %"22", label %"33"]

"22":                                             ; preds = %"16"
  br i1 %0, label %"16", label %"26"

"26":                                             ; preds = %"22"
  unreachable

"33":                                             ; preds = %"16"
  unreachable
}
