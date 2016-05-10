; RUN: opt < %s -memcpyopt -S | FileCheck %s

define void @test_simple_memcpy(i32* noalias %dst, i32* noalias %src) {
; CHECK-LABEL: @test_simple_memcpy
; CHECK-DAG: [[DST:%.*]] = bitcast i32* %dst to i8*
; CHECK-DAG: [[SRC:%.*]] = bitcast i32* %src to i8*
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[DST]], i8* [[SRC]], i64 16, i32 4, i1 false)

  %val.0 = load i32, i32* %src
  store i32 %val.0, i32* %dst

  %src.2 = getelementptr i32, i32* %src, i32 2
  %dst.2 = getelementptr i32, i32* %dst, i32 2
  %val.2 = load i32, i32* %src.2
  store i32 %val.2, i32* %dst.2

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 1
  %val.1 = load i32, i32* %src.1
  store i32 %val.1, i32* %dst.1

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3

  ret void
}

define void @test_simple_memmove(i32* %dst, i32* %src) {
; CHECK-LABEL: @test_simple_memmove
; CHECK-NOT: call void @llvm.memcpy
; CHECK-NOT: call void @llvm.memmove

  %val.0 = load i32, i32* %src
  store i32 %val.0, i32* %dst

  %src.2 = getelementptr i32, i32* %src, i32 2
  %dst.2 = getelementptr i32, i32* %dst, i32 2
  %val.2 = load i32, i32* %src.2
  store i32 %val.2, i32* %dst.2

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 1
  %val.1 = load i32, i32* %src.1
  store i32 %val.1, i32* %dst.1

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3

  ret void
}

; Make sure we can handle calculating bases & offsets from a real memcpy.
define void @test_initial_memcpy(i32* noalias %dst, i32* noalias%src) {
; CHECK-LABEL: @test_initial_memcpy
; CHECK: {{%.*}} = bitcast i32* %dst to i8*
; CHECK: {{%.*}} = bitcast i32* %src to i8*
; CHECK: [[DST:%.*]] = bitcast i32* %dst to i8*
; CHECK: [[SRC:%.*]] = bitcast i32* %src to i8*
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[DST]], i8* [[SRC]], i64 16, i32 4, i1 false)

  %dst.0 = bitcast i32* %dst to i8*
  %src.0 = bitcast i32* %src to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst.0, i8* %src.0, i64 4, i32 4, i1 false)

  %src.2 = getelementptr i32, i32* %src, i32 2
  %dst.2 = getelementptr i32, i32* %dst, i32 2
  %val.2 = load i32, i32* %src.2
  store i32 %val.2, i32* %dst.2

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 1
  %val.1 = load i32, i32* %src.1
  store i32 %val.1, i32* %dst.1

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3

  ret void
}

define void @test_volatile_skipped(i32* noalias %dst, i32* noalias %src) {
; CHECK-LABEL: @test_volatile_skipped
; CHECK-NOT: call void @llvm.memcpy
; CHECK-NOT: call void @llvm.memmove

  %val.0 = load i32, i32* %src
  store i32 %val.0, i32* %dst

  %src.2 = getelementptr i32, i32* %src, i32 2
  %dst.2 = getelementptr i32, i32* %dst, i32 2
  %val.2 = load volatile i32, i32* %src.2
  store i32 %val.2, i32* %dst.2

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 1
  %val.1 = load i32, i32* %src.1
  store i32 %val.1, i32* %dst.1

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3

  ret void
}

define void @test_atomic_skipped(i32* noalias %dst, i32* noalias %src) {
; CHECK-LABEL: @test_atomic_skipped
; CHECK-NOT: call void @llvm.memcpy
; CHECK-NOT: call void @llvm.memmove

  %val.0 = load i32, i32* %src
  store i32 %val.0, i32* %dst

  %src.2 = getelementptr i32, i32* %src, i32 2
  %dst.2 = getelementptr i32, i32* %dst, i32 2
  %val.2 = load i32, i32* %src.2
  store i32 %val.2, i32* %dst.2

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 1
  %val.1 = load i32, i32* %src.1
  store atomic i32 %val.1, i32* %dst.1 unordered, align 4

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3

  ret void
}

define i32 @test_multi_use_skipped(i32* noalias %dst, i32* noalias %src) {
; CHECK-LABEL: @test_multi_use_skipped
; CHECK-NOT: call void @llvm.memcpy
; CHECK-NOT: call void @llvm.memmove

  %val.0 = load i32, i32* %src
  store i32 %val.0, i32* %dst

  %src.2 = getelementptr i32, i32* %src, i32 2
  %dst.2 = getelementptr i32, i32* %dst, i32 2
  %val.2 = load i32, i32* %src.2
  store i32 %val.2, i32* %dst.2

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 1
  %val.1 = load i32, i32* %src.1
  store i32 %val.1, i32* %dst.1

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3

  ret i32 %val.1
}

define void @test_side_effect_skipped(i32* noalias %dst, i32* noalias %src) {
; CHECK-LABEL: @test_side_effect_skipped
; CHECK-NOT: call void @llvm.memcpy
; CHECK-NOT: call void @llvm.memmove

  %val.0 = load i32, i32* %src
  store i32 %val.0, i32* %dst

  %src.2 = getelementptr i32, i32* %src, i32 2
  %dst.2 = getelementptr i32, i32* %dst, i32 2
  %val.2 = load i32, i32* %src.2
  store i32 %val.2, i32* %dst.2

  call void asm sideeffect "", ""()

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 1
  %val.1 = load i32, i32* %src.1
  store i32 %val.1, i32* %dst.1

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3

  ret void
}

define void @test_holes_handled(i32* noalias %dst, i32* noalias %src) {
; CHECK-LABEL: @test_holes_handled
; CHECK-DAG: [[DST:%.*]] = bitcast i32* %dst to i8*
; CHECK-DAG: [[SRC:%.*]] = bitcast i32* %src to i8*
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[DST]], i8* [[SRC]], i64 16, i32 4, i1 false)
; CHECK-DAG: [[DST:%.*]] = bitcast i32* %dst.7 to i8*
; CHECK-DAG: [[SRC:%.*]] = bitcast i32* %src.7 to i8*
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[DST]], i8* [[SRC]], i64 16, i32 4, i1 false)

  %val.0 = load i32, i32* %src
  store i32 %val.0, i32* %dst

  %src.2 = getelementptr i32, i32* %src, i32 2
  %dst.2 = getelementptr i32, i32* %dst, i32 2
  %val.2 = load i32, i32* %src.2
  store i32 %val.2, i32* %dst.2

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 1
  %val.1 = load i32, i32* %src.1
  store i32 %val.1, i32* %dst.1

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3


  %src.7 = getelementptr i32, i32* %src, i32 7
  %dst.7 = getelementptr i32, i32* %dst, i32 7
  %val.7 = load i32, i32* %src.7
  store i32 %val.7, i32* %dst.7

  %src.9 = getelementptr i32, i32* %src, i32 9
  %dst.9 = getelementptr i32, i32* %dst, i32 9
  %val.9 = load i32, i32* %src.9
  store i32 %val.9, i32* %dst.9

  %src.10 = getelementptr i32, i32* %src, i32 10
  %dst.10 = getelementptr i32, i32* %dst, i32 10
  %val.10 = load i32, i32* %src.10
  store i32 %val.10, i32* %dst.10

  %src.8 = getelementptr i32, i32* %src, i32 8
  %dst.8 = getelementptr i32, i32* %dst, i32 8
  %val.8 = load i32, i32* %src.8
  store i32 %val.8, i32* %dst.8

  ret void
}

define void @test_offset_mismatch(i32* noalias %dst, i32* noalias %src) {
; CHECK-LABEL: @test_offset_mismatch
; CHECK-NOT: call void @llvm.memcpy
; CHECK-NOT: call void @llvm.memmove

  %val.0 = load i32, i32* %src
  store i32 %val.0, i32* %dst

  %src.2 = getelementptr i32, i32* %src, i32 2
  %dst.2 = getelementptr i32, i32* %dst, i32 1
  %val.2 = load i32, i32* %src.2
  store i32 %val.2, i32* %dst.2

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 2
  %val.1 = load i32, i32* %src.1
  store i32 %val.1, i32* %dst.1

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3

  ret void
}

define void @test_non_idempotent_ops(i8* %dst, i8* %src) {
; CHECK-LABEL: @test_non_idempotent_ops
; CHECK-NOT: call void @llvm.memcpy
; CHECK-NOT: call void @llvm.memmove

  %val.0 = load i8, i8* %src
  store i8 %val.0, i8* %dst

  %src.2 = getelementptr i8, i8* %src, i8 2
  %dst.2 = getelementptr i8, i8* %dst, i8 2
  %val.2 = load i8, i8* %src.2
  store i8 %val.2, i8* %dst.2

  %val.0.dup = load i8, i8* %src
  store i8 %val.0.dup, i8* %dst

  %src.1 = getelementptr i8, i8* %src, i8 1
  %dst.1 = getelementptr i8, i8* %dst, i8 1
  %val.1 = load i8, i8* %src.1
  store i8 %val.1, i8* %dst.1

  %src.3 = getelementptr i8, i8* %src, i8 3
  %dst.3 = getelementptr i8, i8* %dst, i8 3
  %val.3 = load i8, i8* %src.3
  store i8 %val.3, i8* %dst.3

  ret void
}

define void @test_intervening_op(i32* noalias %dst, i32* noalias %src) {
; CHECK-LABEL: @test_intervening_op
; CHECK-NOT: call void @llvm.memcpy

  %val.0 = load i32, i32* %src
  store i32 %val.0, i32* %dst

  %src.2 = getelementptr i32, i32* %src, i32 2
  %src16.2 = bitcast i32* %src.2 to i16*
  %dst.2 = getelementptr i32, i32* %dst, i32 2
  %val16.2 = load i16, i16* %src16.2
  %val.2 = sext i16 %val16.2 to i32
  store i32 %val.2, i32* %dst.2

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 1
  %val.1 = load i32, i32* %src.1
  store i32 %val.1, i32* %dst.1

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3

  ret void
}

define void @test_infer_align(i32* noalias %dst, i32* noalias %src) {
; CHECK-LABEL: @test_infer_align
; CHECK-DAG: [[DST:%.*]] = bitcast i32* %dst to i8*
; CHECK-DAG: [[SRC:%.*]] = bitcast i32* %src to i8*
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[DST]], i8* [[SRC]], i64 16, i32 8, i1 false)

  %src.2 = getelementptr i32, i32* %src, i32 2
  %dst.2 = getelementptr i32, i32* %dst, i32 2
  %val.2 = load i32, i32* %src.2
  store i32 %val.2, i32* %dst.2

  %val.0 = load i32, i32* %src, align 8
  store i32 %val.0, i32* %dst, align 16

  %src.1 = getelementptr i32, i32* %src, i32 1
  %dst.1 = getelementptr i32, i32* %dst, i32 1
  %val.1 = load i32, i32* %src.1
  store i32 %val.1, i32* %dst.1

  %src.3 = getelementptr i32, i32* %src, i32 3
  %dst.3 = getelementptr i32, i32* %dst, i32 3
  %val.3 = load i32, i32* %src.3
  store i32 %val.3, i32* %dst.3

  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)
