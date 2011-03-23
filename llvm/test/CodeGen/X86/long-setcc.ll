; RUN: llc < %s -march=x86 | FileCheck %s

; General case
define i1 @t1(i64 %x, i64 %y) nounwind {
; CHECK: @t1
; CHECK: subl
; CHECK: sbbl
; CHECK: setl %al
  %B = icmp slt i64 %x, %y
  ret i1 %B
}

; Some special cases
define i1 @t2(i64 %x) nounwind {
; CHECK: @t2
; CHECK: shrl $31, %eax
  %B = icmp slt i64 %x, 0
  ret i1 %B
}

define i1 @t3(i64 %x) nounwind {
; CHECK: @t3
; CHECX: cmpl $0
; CHECX: sete %al
  %tmp = icmp ult i64 %x, 4294967296
  ret i1 %tmp
}

define i1 @t4(i64 %x) nounwind {
; CHECK: @t4
; CHECX: cmpl $0
; CHECX: setne %al
  %tmp = icmp ugt i64 %x, 4294967295
  ret i1 %tmp
}
