; RUN: llc < %s -O0 -fast-isel-abort -mtriple=arm64-apple-darwin | FileCheck %s

define i32 @t1(i32 %c) nounwind readnone {
entry:
; CHECK: @t1
; CHECK: and w0, w0, #0x1
; CHECK: subs w0, w0, #0
; CHECK: csel w0, w{{[0-9]+}}, w{{[0-9]+}}, ne
  %0 = icmp sgt i32 %c, 1
  %1 = select i1 %0, i32 123, i32 357
  ret i32 %1
}

define i64 @t2(i32 %c) nounwind readnone {
entry:
; CHECK: @t2
; CHECK: and w0, w0, #0x1
; CHECK: subs w0, w0, #0
; CHECK: csel x0, x{{[0-9]+}}, x{{[0-9]+}}, ne
  %0 = icmp sgt i32 %c, 1
  %1 = select i1 %0, i64 123, i64 357
  ret i64 %1
}

define i32 @t3(i1 %c, i32 %a, i32 %b) nounwind readnone {
entry:
; CHECK: @t3
; CHECK: and w0, w0, #0x1
; CHECK: subs w0, w0, #0
; CHECK: csel w0, w{{[0-9]+}}, w{{[0-9]+}}, ne
  %0 = select i1 %c, i32 %a, i32 %b
  ret i32 %0
}

define i64 @t4(i1 %c, i64 %a, i64 %b) nounwind readnone {
entry:
; CHECK: @t4
; CHECK: and w0, w0, #0x1
; CHECK: subs w0, w0, #0
; CHECK: csel x0, x{{[0-9]+}}, x{{[0-9]+}}, ne
  %0 = select i1 %c, i64 %a, i64 %b
  ret i64 %0
}

define float @t5(i1 %c, float %a, float %b) nounwind readnone {
entry:
; CHECK: @t5
; CHECK: and w0, w0, #0x1
; CHECK: subs w0, w0, #0
; CHECK: fcsel s0, s0, s1, ne
  %0 = select i1 %c, float %a, float %b
  ret float %0
}

define double @t6(i1 %c, double %a, double %b) nounwind readnone {
entry:
; CHECK: @t6
; CHECK: and w0, w0, #0x1
; CHECK: subs w0, w0, #0
; CHECK: fcsel d0, d0, d1, ne
  %0 = select i1 %c, double %a, double %b
  ret double %0
}
