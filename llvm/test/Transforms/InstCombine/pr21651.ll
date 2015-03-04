; RUN: opt < %s -instcombine -S | FileCheck %s

; Provide legal integer types.
target datalayout = "n8:16:32:64"


define void @PR21651() {
  switch i2 0, label %out [
    i2 0, label %out
    i2 1, label %out
  ]

out:
  ret void
}

; CHECK-LABEL: define void @PR21651(
; CHECK:   switch i2 0, label %out [
; CHECK:     i2 0, label %out
; CHECK:     i2 1, label %out
; CHECK:   ]
; CHECK: out:                                              ; preds = %0, %0, %0
; CHECK:   ret void
; CHECK: }
