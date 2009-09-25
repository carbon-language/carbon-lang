; RUN: llc < %s -march=x86-64 -asm-verbose=false | FileCheck %s

; Currently, floating-point selects are lowered to CFG triangles.
; This means that one side of the select is always unconditionally
; evaluated, however with MachineSink we can sink the other side so
; that it's conditionally evaluated.

; CHECK: foo:
; CHECK-NEXT: divsd
; CHECK-NEXT: testb $1, %dil
; CHECK-NEXT: jne

define double @foo(double %x, double %y, i1 %c) nounwind {
  %a = fdiv double %x, 3.2
  %b = fdiv double %y, 3.3
  %z = select i1 %c, double %a, double %b
  ret double %z
}

; Hoist floating-point constant-pool loads out of loops.

; CHECK: bar:
; CHECK: movsd
; CHECK: align
define void @bar(double* nocapture %p, i64 %n) nounwind {
entry:
  %0 = icmp sgt i64 %n, 0
  br i1 %0, label %bb, label %return

bb:
  %i.03 = phi i64 [ 0, %entry ], [ %3, %bb ]
  %scevgep = getelementptr double* %p, i64 %i.03
  %1 = load double* %scevgep, align 8
  %2 = fdiv double 3.200000e+00, %1
  store double %2, double* %scevgep, align 8
  %3 = add nsw i64 %i.03, 1
  %exitcond = icmp eq i64 %3, %n
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}
