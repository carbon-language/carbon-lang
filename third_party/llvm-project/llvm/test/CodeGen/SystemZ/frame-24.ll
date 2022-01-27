; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
;
; Test saving of vararg registers and backchain with packed stack.

%struct.__va_list_tag = type { i64, i64, i8*, i8* }
declare void @llvm.va_start(i8*)

attributes #0 = { nounwind "packed-stack"="true" }
define void @fun0(i64 %g0, double %d0, i64 %n, ...) #0 {
; CHECK-LABEL: fun0:
; CHECK:      stmg	%r4, %r15, 32(%r15)
; CHECK-NEXT: aghi	%r15, -192
; CHECK-NEXT: std	%f2, 328(%r15)
; CHECK-NEXT: std	%f4, 336(%r15)
; CHECK-NEXT: std	%f6, 344(%r15)
; CHECK-NEXT: la	%r0, 352(%r15)
; CHECK-NEXT: stg	%r0, 176(%r15)
; CHECK-NEXT: la	%r0, 192(%r15)
; CHECK-NEXT: stg	%r0, 184(%r15)
; CHECK-NEXT: mvghi	160(%r15), 2
; CHECK-NEXT: mvghi	168(%r15), 1
; CHECK-NEXT: lmg	%r6, %r15, 240(%r15)
; CHECK-NEXT: br	%r14
entry:
  %vl = alloca [1 x %struct.__va_list_tag], align 8
  %0 = bitcast [1 x %struct.__va_list_tag]* %vl to i8*
  call void @llvm.va_start(i8* nonnull %0)
  ret void
}

attributes #1 = { nounwind "packed-stack"="true" "use-soft-float"="true" }
define void @fun1(i64 %g0, double %d0, i64 %n, ...) #1 {
; CHECK-LABEL: fun1:
; CHECK:      stmg	%r5, %r15, 72(%r15)
; CHECK-NEXT: aghi	%r15, -160
; CHECK-NEXT: la	%r0, 192(%r15)
; CHECK-NEXT: stg	%r0, 184(%r15)
; CHECK-NEXT: la	%r0, 320(%r15)
; CHECK-NEXT: stg	%r0, 176(%r15)
; CHECK-NEXT: mvghi	168(%r15), 0
; CHECK-NEXT: mvghi	160(%r15), 3
; CHECK-NEXT: lmg	%r6, %r15, 240(%r15)
; CHECK-NEXT: br	%r14
entry:
  %vl = alloca [1 x %struct.__va_list_tag], align 8
  %0 = bitcast [1 x %struct.__va_list_tag]* %vl to i8*
  call void @llvm.va_start(i8* nonnull %0)
  ret void
}

attributes #2 = { nounwind "packed-stack"="true" "use-soft-float"="true" "backchain"}
define void @fun2(i64 %g0, double %d0, i64 %n, ...) #2 {
; CHECK-LABEL: fun2:
; CHECK:      stmg	%r5, %r15, 64(%r15)
; CHECK-NEXT: lgr	%r1, %r15
; CHECK-NEXT: aghi	%r15, -168
; CHECK-NEXT: stg	%r1, 152(%r15)
; CHECK-NEXT: la	%r0, 192(%r15)
; CHECK-NEXT: stg	%r0, 184(%r15)
; CHECK-NEXT: la	%r0, 328(%r15)
; CHECK-NEXT: stg	%r0, 176(%r15)
; CHECK-NEXT: mvghi	168(%r15), 0
; CHECK-NEXT: mvghi	160(%r15), 3
; CHECK-NEXT: lmg	%r6, %r15, 240(%r15)
; CHECK-NEXT: br	%r14
entry:
  %vl = alloca [1 x %struct.__va_list_tag], align 8
  %0 = bitcast [1 x %struct.__va_list_tag]* %vl to i8*
  call void @llvm.va_start(i8* nonnull %0)
  ret void
}

