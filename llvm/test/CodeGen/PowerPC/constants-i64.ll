; RUN: llc -mcpu=ppc64 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define i64 @cn1() #0 {
entry:
  ret i64 281474976710655

; CHECK-LABEL: @cn1
; CHECK: li [[REG1:[0-9]+]], 0
; CHECK: ori [[REG2:[0-9]+]], [[REG1]], 65535
; CHECK: sldi [[REG3:[0-9]+]], [[REG2]], 48
; CHECK: nor 3, [[REG3]], [[REG3]]
; CHECK: blr
}

attributes #0 = { nounwind readnone }


