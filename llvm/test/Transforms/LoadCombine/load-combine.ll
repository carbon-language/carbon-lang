; RUN: opt < %s -load-combine -instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Combine read from char* idiom.
define i64 @LoadU64_x64_0(i64* %pData) {
  %1 = bitcast i64* %pData to i8*
  %2 = load i8* %1, align 1
  %3 = zext i8 %2 to i64
  %4 = shl nuw i64 %3, 56
  %5 = getelementptr inbounds i8* %1, i64 1
  %6 = load i8* %5, align 1
  %7 = zext i8 %6 to i64
  %8 = shl nuw nsw i64 %7, 48
  %9 = or i64 %8, %4
  %10 = getelementptr inbounds i8* %1, i64 2
  %11 = load i8* %10, align 1
  %12 = zext i8 %11 to i64
  %13 = shl nuw nsw i64 %12, 40
  %14 = or i64 %9, %13
  %15 = getelementptr inbounds i8* %1, i64 3
  %16 = load i8* %15, align 1
  %17 = zext i8 %16 to i64
  %18 = shl nuw nsw i64 %17, 32
  %19 = or i64 %14, %18
  %20 = getelementptr inbounds i8* %1, i64 4
  %21 = load i8* %20, align 1
  %22 = zext i8 %21 to i64
  %23 = shl nuw nsw i64 %22, 24
  %24 = or i64 %19, %23
  %25 = getelementptr inbounds i8* %1, i64 5
  %26 = load i8* %25, align 1
  %27 = zext i8 %26 to i64
  %28 = shl nuw nsw i64 %27, 16
  %29 = or i64 %24, %28
  %30 = getelementptr inbounds i8* %1, i64 6
  %31 = load i8* %30, align 1
  %32 = zext i8 %31 to i64
  %33 = shl nuw nsw i64 %32, 8
  %34 = or i64 %29, %33
  %35 = getelementptr inbounds i8* %1, i64 7
  %36 = load i8* %35, align 1
  %37 = zext i8 %36 to i64
  %38 = or i64 %34, %37
  ret i64 %38
; CHECK-LABEL: @LoadU64_x64_0(
; CHECK: load i64* %{{.*}}, align 1
; CHECK-NOT: load
}

; Combine simple adjacent loads.
define i32 @"2xi16_i32"(i16* %x) {
  %1 = load i16* %x, align 2
  %2 = getelementptr inbounds i16* %x, i64 1
  %3 = load i16* %2, align 2
  %4 = zext i16 %3 to i32
  %5 = shl nuw i32 %4, 16
  %6 = zext i16 %1 to i32
  %7 = or i32 %5, %6
  ret i32 %7
; CHECK-LABEL: @"2xi16_i32"(
; CHECK: load i32* %{{.*}}, align 2
; CHECK-NOT: load
}

; Don't combine loads across stores.
define i32 @"2xi16_i32_store"(i16* %x, i16* %y) {
  %1 = load i16* %x, align 2
  store i16 0, i16* %y, align 2
  %2 = getelementptr inbounds i16* %x, i64 1
  %3 = load i16* %2, align 2
  %4 = zext i16 %3 to i32
  %5 = shl nuw i32 %4, 16
  %6 = zext i16 %1 to i32
  %7 = or i32 %5, %6
  ret i32 %7
; CHECK-LABEL: @"2xi16_i32_store"(
; CHECK: load i16* %{{.*}}, align 2
; CHECK: store
; CHECK: load i16* %{{.*}}, align 2
}

; Don't combine loads with a gap.
define i32 @"2xi16_i32_gap"(i16* %x) {
  %1 = load i16* %x, align 2
  %2 = getelementptr inbounds i16* %x, i64 2
  %3 = load i16* %2, align 2
  %4 = zext i16 %3 to i32
  %5 = shl nuw i32 %4, 16
  %6 = zext i16 %1 to i32
  %7 = or i32 %5, %6
  ret i32 %7
; CHECK-LABEL: @"2xi16_i32_gap"(
; CHECK: load i16* %{{.*}}, align 2
; CHECK: load i16* %{{.*}}, align 2
}

; Combine out of order loads.
define i32 @"2xi16_i32_order"(i16* %x) {
  %1 = getelementptr inbounds i16* %x, i64 1
  %2 = load i16* %1, align 2
  %3 = zext i16 %2 to i32
  %4 = load i16* %x, align 2
  %5 = shl nuw i32 %3, 16
  %6 = zext i16 %4 to i32
  %7 = or i32 %5, %6
  ret i32 %7
; CHECK-LABEL: @"2xi16_i32_order"(
; CHECK: load i32* %{{.*}}, align 2
; CHECK-NOT: load
}

; Overlapping loads.
define i32 @"2xi16_i32_overlap"(i8* %x) {
  %1 = bitcast i8* %x to i16*
  %2 = load i16* %1, align 2
  %3 = getelementptr inbounds i8* %x, i64 1
  %4 = bitcast i8* %3 to i16*
  %5 = load i16* %4, align 2
  %6 = zext i16 %5 to i32
  %7 = shl nuw i32 %6, 16
  %8 = zext i16 %2 to i32
  %9 = or i32 %7, %8
  ret i32 %9
; CHECK-LABEL: @"2xi16_i32_overlap"(
; CHECK: load i16* %{{.*}}, align 2
; CHECK: load i16* %{{.*}}, align 2
}

; Combine valid alignments.
define i64 @"2xi16_i64_align"(i8* %x) {
  %1 = bitcast i8* %x to i32*
  %2 = load i32* %1, align 4
  %3 = getelementptr inbounds i8* %x, i64 4
  %4 = bitcast i8* %3 to i16*
  %5 = load i16* %4, align 2
  %6 = getelementptr inbounds i8* %x, i64 6
  %7 = bitcast i8* %6 to i16*
  %8 = load i16* %7, align 2
  %9 = zext i16 %8 to i64
  %10 = shl nuw i64 %9, 48
  %11 = zext i16 %5 to i64
  %12 = shl nuw nsw i64 %11, 32
  %13 = zext i32 %2 to i64
  %14 = or i64 %12, %13
  %15 = or i64 %14, %10
  ret i64 %15
; CHECK-LABEL: @"2xi16_i64_align"(
; CHECK: load i64* %{{.*}}, align 4
}

; Non power of two.
define i64 @"2xi16_i64_npo2"(i8* %x) {
  %1 = load i8* %x, align 1
  %2 = zext i8 %1 to i64
  %3 = getelementptr inbounds i8* %x, i64 1
  %4 = load i8* %3, align 1
  %5 = zext i8 %4 to i64
  %6 = shl nuw nsw i64 %5, 8
  %7 = or i64 %6, %2
  %8 = getelementptr inbounds i8* %x, i64 2
  %9 = load i8* %8, align 1
  %10 = zext i8 %9 to i64
  %11 = shl nuw nsw i64 %10, 16
  %12 = or i64 %11, %7
  %13 = getelementptr inbounds i8* %x, i64 3
  %14 = load i8* %13, align 1
  %15 = zext i8 %14 to i64
  %16 = shl nuw nsw i64 %15, 24
  %17 = or i64 %16, %12
  %18 = getelementptr inbounds i8* %x, i64 4
  %19 = load i8* %18, align 1
  %20 = zext i8 %19 to i64
  %21 = shl nuw nsw i64 %20, 32
  %22 = or i64 %21, %17
  %23 = getelementptr inbounds i8* %x, i64 5
  %24 = load i8* %23, align 1
  %25 = zext i8 %24 to i64
  %26 = shl nuw nsw i64 %25, 40
  %27 = or i64 %26, %22
  %28 = getelementptr inbounds i8* %x, i64 6
  %29 = load i8* %28, align 1
  %30 = zext i8 %29 to i64
  %31 = shl nuw nsw i64 %30, 48
  %32 = or i64 %31, %27
  ret i64 %32
; CHECK-LABEL: @"2xi16_i64_npo2"(
; CHECK: load i32* %{{.*}}, align 1
}
