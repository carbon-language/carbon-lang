; Test load/store pairs that act as memcpys.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g1 = global i8 1
@g2 = global i16 2
@g3 = global i32 3
@g4 = global i64 4
@g5 = external global fp128, align 16

; Test the simple i8 case.
define void @f1(i8 *%ptr1) {
; CHECK-LABEL: f1:
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8 *%ptr1, i64 1
  %val = load i8 *%ptr1
  store i8 %val, i8 *%ptr2
  ret void
}

; Test i8 cases where the value is zero-extended to 32 bits.
define void @f2(i8 *%ptr1) {
; CHECK-LABEL: f2:
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8 *%ptr1, i64 1
  %val = load i8 *%ptr1
  %ext = zext i8 %val to i32
  %trunc = trunc i32 %ext to i8
  store i8 %trunc, i8 *%ptr2
  ret void
}

; Test i8 cases where the value is zero-extended to 64 bits.
define void @f3(i8 *%ptr1) {
; CHECK-LABEL: f3:
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8 *%ptr1, i64 1
  %val = load i8 *%ptr1
  %ext = zext i8 %val to i64
  %trunc = trunc i64 %ext to i8
  store i8 %trunc, i8 *%ptr2
  ret void
}

; Test i8 cases where the value is sign-extended to 32 bits.
define void @f4(i8 *%ptr1) {
; CHECK-LABEL: f4:
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8 *%ptr1, i64 1
  %val = load i8 *%ptr1
  %ext = sext i8 %val to i32
  %trunc = trunc i32 %ext to i8
  store i8 %trunc, i8 *%ptr2
  ret void
}

; Test i8 cases where the value is sign-extended to 64 bits.
define void @f5(i8 *%ptr1) {
; CHECK-LABEL: f5:
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8 *%ptr1, i64 1
  %val = load i8 *%ptr1
  %ext = sext i8 %val to i64
  %trunc = trunc i64 %ext to i8
  store i8 %trunc, i8 *%ptr2
  ret void
}

; Test the simple i16 case.
define void @f6(i16 *%ptr1) {
; CHECK-LABEL: f6:
; CHECK: mvc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16 *%ptr1, i64 1
  %val = load i16 *%ptr1
  store i16 %val, i16 *%ptr2
  ret void
}

; Test i16 cases where the value is zero-extended to 32 bits.
define void @f7(i16 *%ptr1) {
; CHECK-LABEL: f7:
; CHECK: mvc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16 *%ptr1, i64 1
  %val = load i16 *%ptr1
  %ext = zext i16 %val to i32
  %trunc = trunc i32 %ext to i16
  store i16 %trunc, i16 *%ptr2
  ret void
}

; Test i16 cases where the value is zero-extended to 64 bits.
define void @f8(i16 *%ptr1) {
; CHECK-LABEL: f8:
; CHECK: mvc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16 *%ptr1, i64 1
  %val = load i16 *%ptr1
  %ext = zext i16 %val to i64
  %trunc = trunc i64 %ext to i16
  store i16 %trunc, i16 *%ptr2
  ret void
}

; Test i16 cases where the value is sign-extended to 32 bits.
define void @f9(i16 *%ptr1) {
; CHECK-LABEL: f9:
; CHECK: mvc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16 *%ptr1, i64 1
  %val = load i16 *%ptr1
  %ext = sext i16 %val to i32
  %trunc = trunc i32 %ext to i16
  store i16 %trunc, i16 *%ptr2
  ret void
}

; Test i16 cases where the value is sign-extended to 64 bits.
define void @f10(i16 *%ptr1) {
; CHECK-LABEL: f10:
; CHECK: mvc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16 *%ptr1, i64 1
  %val = load i16 *%ptr1
  %ext = sext i16 %val to i64
  %trunc = trunc i64 %ext to i16
  store i16 %trunc, i16 *%ptr2
  ret void
}

; Test the simple i32 case.
define void @f11(i32 *%ptr1) {
; CHECK-LABEL: f11:
; CHECK: mvc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i32 *%ptr1, i64 1
  %val = load i32 *%ptr1
  store i32 %val, i32 *%ptr2
  ret void
}

; Test i32 cases where the value is zero-extended to 64 bits.
define void @f12(i32 *%ptr1) {
; CHECK-LABEL: f12:
; CHECK: mvc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i32 *%ptr1, i64 1
  %val = load i32 *%ptr1
  %ext = zext i32 %val to i64
  %trunc = trunc i64 %ext to i32
  store i32 %trunc, i32 *%ptr2
  ret void
}

; Test i32 cases where the value is sign-extended to 64 bits.
define void @f13(i32 *%ptr1) {
; CHECK-LABEL: f13:
; CHECK: mvc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i32 *%ptr1, i64 1
  %val = load i32 *%ptr1
  %ext = sext i32 %val to i64
  %trunc = trunc i64 %ext to i32
  store i32 %trunc, i32 *%ptr2
  ret void
}

; Test the i64 case.
define void @f14(i64 *%ptr1) {
; CHECK-LABEL: f14:
; CHECK: mvc 8(8,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i64 *%ptr1, i64 1
  %val = load i64 *%ptr1
  store i64 %val, i64 *%ptr2
  ret void
}

; Test the f32 case.
define void @f15(float *%ptr1) {
; CHECK-LABEL: f15:
; CHECK: mvc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr float *%ptr1, i64 1
  %val = load float *%ptr1
  store float %val, float *%ptr2
  ret void
}

; Test the f64 case.
define void @f16(double *%ptr1) {
; CHECK-LABEL: f16:
; CHECK: mvc 8(8,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr double *%ptr1, i64 1
  %val = load double *%ptr1
  store double %val, double *%ptr2
  ret void
}

; Test the f128 case.
define void @f17(fp128 *%ptr1) {
; CHECK-LABEL: f17:
; CHECK: mvc 16(16,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr fp128 *%ptr1, i64 1
  %val = load fp128 *%ptr1
  store fp128 %val, fp128 *%ptr2
  ret void
}

; Make sure that we don't use MVC if the load is volatile.
define void @f18(i64 *%ptr1) {
; CHECK-LABEL: f18:
; CHECK-NOT: mvc
; CHECK: br %r14
  %ptr2 = getelementptr i64 *%ptr1, i64 1
  %val = load volatile i64 *%ptr1
  store i64 %val, i64 *%ptr2
  ret void
}

; ...likewise the store.
define void @f19(i64 *%ptr1) {
; CHECK-LABEL: f19:
; CHECK-NOT: mvc
; CHECK: br %r14
  %ptr2 = getelementptr i64 *%ptr1, i64 1
  %val = load i64 *%ptr1
  store volatile i64 %val, i64 *%ptr2
  ret void
}

; Test that MVC is used for aligned loads and stores, even if there is
; no way of telling whether they alias.
define void @f20(i64 *%ptr1, i64 *%ptr2) {
; CHECK-LABEL: f20:
; CHECK: mvc 0(8,%r3), 0(%r2)
; CHECK: br %r14
  %val = load i64 *%ptr1
  store i64 %val, i64 *%ptr2
  ret void
}

; ...but if the loads aren't aligned, we can't be sure.
define void @f21(i64 *%ptr1, i64 *%ptr2) {
; CHECK-LABEL: f21:
; CHECK-NOT: mvc
; CHECK: br %r14
  %val = load i64 *%ptr1, align 2
  store i64 %val, i64 *%ptr2, align 2
  ret void
}

; Test a case where there is definite overlap.
define void @f22(i64 %base) {
; CHECK-LABEL: f22:
; CHECK-NOT: mvc
; CHECK: br %r14
  %add = add i64 %base, 1
  %ptr1 = inttoptr i64 %base to i64 *
  %ptr2 = inttoptr i64 %add to i64 *
  %val = load i64 *%ptr1, align 1
  store i64 %val, i64 *%ptr2, align 1
  ret void
}

; Test that we can use MVC for global addresses for i8.
define void @f23(i8 *%ptr) {
; CHECK-LABEL: f23:
; CHECK: larl [[REG:%r[0-5]]], g1
; CHECK: mvc 0(1,%r2), 0([[REG]])
; CHECK: br %r14
  %val = load i8 *@g1
  store i8 %val, i8 *%ptr
  ret void
}

; ...and again with the global on the store.
define void @f24(i8 *%ptr) {
; CHECK-LABEL: f24:
; CHECK: larl [[REG:%r[0-5]]], g1
; CHECK: mvc 0(1,[[REG]]), 0(%r2)
; CHECK: br %r14
  %val = load i8 *%ptr
  store i8 %val, i8 *@g1
  ret void
}

; Test that we use LHRL for i16.
define void @f25(i16 *%ptr) {
; CHECK-LABEL: f25:
; CHECK: lhrl [[REG:%r[0-5]]], g2
; CHECK: sth [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i16 *@g2
  store i16 %val, i16 *%ptr
  ret void
}

; ...likewise STHRL.
define void @f26(i16 *%ptr) {
; CHECK-LABEL: f26:
; CHECK: lh [[REG:%r[0-5]]], 0(%r2)
; CHECK: sthrl [[REG]], g2
; CHECK: br %r14
  %val = load i16 *%ptr
  store i16 %val, i16 *@g2
  ret void
}

; Test that we use LRL for i32.
define void @f27(i32 *%ptr) {
; CHECK-LABEL: f27:
; CHECK: lrl [[REG:%r[0-5]]], g3
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i32 *@g3
  store i32 %val, i32 *%ptr
  ret void
}

; ...likewise STRL.
define void @f28(i32 *%ptr) {
; CHECK-LABEL: f28:
; CHECK: l [[REG:%r[0-5]]], 0(%r2)
; CHECK: strl [[REG]], g3
; CHECK: br %r14
  %val = load i32 *%ptr
  store i32 %val, i32 *@g3
  ret void
}

; Test that we use LGRL for i64.
define void @f29(i64 *%ptr) {
; CHECK-LABEL: f29:
; CHECK: lgrl [[REG:%r[0-5]]], g4
; CHECK: stg [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i64 *@g4
  store i64 %val, i64 *%ptr
  ret void
}

; ...likewise STGRL.
define void @f30(i64 *%ptr) {
; CHECK-LABEL: f30:
; CHECK: lg [[REG:%r[0-5]]], 0(%r2)
; CHECK: stgrl [[REG]], g4
; CHECK: br %r14
  %val = load i64 *%ptr
  store i64 %val, i64 *@g4
  ret void
}

; Test that we can use MVC for global addresses for fp128.
define void @f31(fp128 *%ptr) {
; CHECK-LABEL: f31:
; CHECK: larl [[REG:%r[0-5]]], g5
; CHECK: mvc 0(16,%r2), 0([[REG]])
; CHECK: br %r14
  %val = load fp128 *@g5, align 16
  store fp128 %val, fp128 *%ptr, align 16
  ret void
}

; ...and again with the global on the store.
define void @f32(fp128 *%ptr) {
; CHECK-LABEL: f32:
; CHECK: larl [[REG:%r[0-5]]], g5
; CHECK: mvc 0(16,[[REG]]), 0(%r2)
; CHECK: br %r14
  %val = load fp128 *%ptr, align 16
  store fp128 %val, fp128 *@g5, align 16
  ret void
}

; Test a case where offset disambiguation is enough.
define void @f33(i64 *%ptr1) {
; CHECK-LABEL: f33:
; CHECK: mvc 8(8,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i64 *%ptr1, i64 1
  %val = load i64 *%ptr1, align 1
  store i64 %val, i64 *%ptr2, align 1
  ret void
}

; Test f21 in cases where TBAA tells us there is no alias.
define void @f34(i64 *%ptr1, i64 *%ptr2) {
; CHECK-LABEL: f34:
; CHECK: mvc 0(8,%r3), 0(%r2)
; CHECK: br %r14
  %val = load i64 *%ptr1, align 2, !tbaa !1
  store i64 %val, i64 *%ptr2, align 2, !tbaa !2
  ret void
}

; Test f21 in cases where TBAA is present but doesn't help.
define void @f35(i64 *%ptr1, i64 *%ptr2) {
; CHECK-LABEL: f35:
; CHECK-NOT: mvc
; CHECK: br %r14
  %val = load i64 *%ptr1, align 2, !tbaa !1
  store i64 %val, i64 *%ptr2, align 2, !tbaa !1
  ret void
}

!0 = metadata !{ metadata !"root" }
!1 = metadata !{ metadata !"set1", metadata !0 }
!2 = metadata !{ metadata !"set2", metadata !0 }
