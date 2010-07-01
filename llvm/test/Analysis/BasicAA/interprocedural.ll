; RUN: opt -interprocedural-basic-aa  -interprocedural-aa-eval -print-all-alias-modref-info -disable-output < %s |& FileCheck %s

; The noalias attribute is not safe in an interprocedural context.
; CHECK: MayAlias: i8* %p, i8* %q

define void @t0(i8* noalias %p) {
  store i8 0, i8* %p
  ret void
}
define void @t1(i8* noalias %q) {
  store i8 0, i8* %q
  ret void
}

; An alloca can alias an argument in a different function.
; CHECK: MayAlias: i32* %r, i32* %s

define void @s0(i32* %r) {
  store i32 0, i32* %r
  ret void
}

define void @s1() {
  %s = alloca i32, i32 10
  store i32 0, i32* %s
  call void @s0(i32* %s)
  ret void
}

; An alloca does not alias an argument in the same function.
; CHECK: NoAlias: i64* %t, i64* %u
; CHECK: NoAlias: i64* %a, i64* %u
; CHECK: NoAlias: i64* %a, i64* %t
; CHECK: MayAlias: i64* %u, i64* %v
; CHECK: MayAlias: i64* %t, i64* %v
; CHECK: NoAlias: i64* %a, i64* %v
; CHECK: MayAlias: i64* %b, i64* %u
; CHECK: MayAlias: i64* %b, i64* %t
; CHECK: MayAlias: i64* %b, i64* %v
declare i64* @r0_callee(i64*)
define i64* @r0(i64* %u) {
  %t = alloca i64, i32 10
  %a = alloca i64, i32 10
  %v = call i64* @r0_callee(i64* %t)
  %b = call i64* @r0_callee(i64* %t)
  store i64 0, i64* %t
  store i64 0, i64* %u
  store i64 0, i64* %v
  store i64 0, i64* %a
  store i64 0, i64* %b
  ret i64* %t
}

; The noalias attribute is safe when both arguments belong to the same function
; even in an interprocedural context.
; CHECK: NoAlias: i8* %w, i8* %x

define void @q0(i8* noalias %w, i8* noalias %x) {
  store i8 0, i8* %w
  store i8 0, i8* %x
  call void @q0(i8* noalias %x, i8* noalias %w)
  unreachable
}

; The noalias attribute is not necessarily safe in an interprocedural context.
; CHECK: MayAlias: double* %y, double* @G

@G = external global double
define void @p0(double* noalias %y) {
  store double 0.0, double* %y
  store double 0.0, double* @G
  unreachable
}
