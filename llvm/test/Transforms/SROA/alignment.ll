; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1)

define void @test1({ i8, i8 }* %a, { i8, i8 }* %b) {
; CHECK: @test1
; CHECK: %[[gep_a0:.*]] = getelementptr inbounds { i8, i8 }* %a, i64 0, i32 0
; CHECK: %[[a0:.*]] = load i8* %[[gep_a0]], align 16
; CHECK: %[[gep_a1:.*]] = getelementptr inbounds { i8, i8 }* %a, i64 0, i32 1
; CHECK: %[[a1:.*]] = load i8* %[[gep_a1]], align 1
; CHECK: %[[gep_b0:.*]] = getelementptr inbounds { i8, i8 }* %b, i64 0, i32 0
; CHECK: store i8 %[[a0]], i8* %[[gep_b0]], align 16
; CHECK: %[[gep_b1:.*]] = getelementptr inbounds { i8, i8 }* %b, i64 0, i32 1
; CHECK: store i8 %[[a1]], i8* %[[gep_b1]], align 1
; CHECK: ret void

entry:
  %alloca = alloca { i8, i8 }, align 16
  %gep_a = getelementptr { i8, i8 }* %a, i32 0, i32 0
  %gep_alloca = getelementptr { i8, i8 }* %alloca, i32 0, i32 0
  %gep_b = getelementptr { i8, i8 }* %b, i32 0, i32 0

  store i8 420, i8* %gep_alloca, align 16

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %gep_alloca, i8* %gep_a, i32 2, i32 16, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %gep_b, i8* %gep_alloca, i32 2, i32 16, i1 false)
  ret void
}

define void @test2() {
; CHECK: @test2
; CHECK: alloca i16
; CHECK: load i8* %{{.*}}, align 1
; CHECK: store i8 42, i8* %{{.*}}, align 1
; CHECK: ret void

entry:
  %a = alloca { i8, i8, i8, i8 }, align 2
  %gep1 = getelementptr { i8, i8, i8, i8 }* %a, i32 0, i32 1
  %cast1 = bitcast i8* %gep1 to i16*
  store volatile i16 0, i16* %cast1
  %gep2 = getelementptr { i8, i8, i8, i8 }* %a, i32 0, i32 2
  %result = load i8* %gep2, align 2
  store i8 42, i8* %gep2, align 2
  ret void
}

define void @PR13920(<2 x i64>* %a, i16* %b) {
; Test that alignments on memcpy intrinsics get propagated to loads and stores.
; CHECK: @PR13920
; CHECK: load <2 x i64>* %a, align 2
; CHECK: store <2 x i64> {{.*}}, <2 x i64>* {{.*}}, align 2
; CHECK: ret void

entry:
  %aa = alloca <2 x i64>, align 16
  %aptr = bitcast <2 x i64>* %a to i8*
  %aaptr = bitcast <2 x i64>* %aa to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %aaptr, i8* %aptr, i32 16, i32 2, i1 false)
  %bptr = bitcast i16* %b to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %bptr, i8* %aaptr, i32 16, i32 2, i1 false)
  ret void
}

define void @test3(i8* %x) {
; Test that when we promote an alloca to a type with lower ABI alignment, we
; provide the needed explicit alignment that code using the alloca may be
; expecting. However, also check that any offset within an alloca can in turn
; reduce the alignment.
; CHECK: @test3
; CHECK: alloca [22 x i8], align 8
; CHECK: alloca [18 x i8], align 2
; CHECK: ret void

entry:
  %a = alloca { i8*, i8*, i8* }
  %b = alloca { i8*, i8*, i8* }
  %a_raw = bitcast { i8*, i8*, i8* }* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a_raw, i8* %x, i32 22, i32 8, i1 false)
  %b_raw = bitcast { i8*, i8*, i8* }* %b to i8*
  %b_gep = getelementptr i8* %b_raw, i32 6
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %b_gep, i8* %x, i32 18, i32 2, i1 false)
  ret void
}
