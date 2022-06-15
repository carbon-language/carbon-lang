; RUN: mlir-translate -import-llvm %s | FileCheck %s

define void @intrinsics() {
  ret void
}

; CHECK-LABEL:  llvm.func @fmuladd_test
define void @fmuladd_test(float %0, float %1, <8 x float> %2, i8* %3) {
  ; CHECK: "llvm.intr.fmuladd"(%{{.*}}, %{{.*}}, %{{.*}}) : (f32, f32, f32) -> f32
  %5 = call float @llvm.fmuladd.f32(float %0, float %1, float %0)
  ; CHECK: "llvm.intr.fmuladd"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %6 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %2, <8 x float> %2, <8 x float> %2)
  ; CHECK: "llvm.intr.fma"(%{{.*}}, %{{.*}}, %{{.*}}) : (f32, f32, f32) -> f32
  %7 = call float @llvm.fma.f32(float %0, float %1, float %0)
  ; CHECK: "llvm.intr.fma"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %8 = call <8 x float> @llvm.fma.v8f32(<8 x float> %2, <8 x float> %2, <8 x float> %2)
  ; CHECK: "llvm.intr.prefetch"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, i32, i32, i32) -> ()
  call void @llvm.prefetch.p0i8(i8* %3, i32 0, i32 3, i32 1)
  ret void
}

; CHECK-LABEL:  llvm.func @exp_test
define void @exp_test(float %0, <8 x float> %1) {
  ; CHECK: "llvm.intr.exp"(%{{.*}}) : (f32) -> f32
  %3 = call float @llvm.exp.f32(float %0)
  ; CHECK: "llvm.intr.exp"(%{{.*}}) : (vector<8xf32>) -> vector<8xf32>
  %4 = call <8 x float> @llvm.exp.v8f32(<8 x float> %1)
  ret void
}

; CHECK-LABEL:  llvm.func @exp2_test
define void @exp2_test(float %0, <8 x float> %1) {
  ; CHECK:  "llvm.intr.exp2"(%{{.*}}) : (f32) -> f32
  %3 = call float @llvm.exp2.f32(float %0)
  ; CHECK:  "llvm.intr.exp2"(%{{.*}}) : (vector<8xf32>) -> vector<8xf32>
  %4 = call <8 x float> @llvm.exp2.v8f32(<8 x float> %1)
  ret void
}

; CHECK-LABEL:  llvm.func @log_test
define void @log_test(float %0, <8 x float> %1) {
  ; CHECK:  "llvm.intr.log"(%{{.*}}) : (f32) -> f32
  %3 = call float @llvm.log.f32(float %0)
  ; CHECK:  "llvm.intr.log"(%{{.*}}) : (vector<8xf32>) -> vector<8xf32>
  %4 = call <8 x float> @llvm.log.v8f32(<8 x float> %1)
  ret void
}

; CHECK-LABEL:  llvm.func @log10_test
define void @log10_test(float %0, <8 x float> %1) {
  ; CHECK:  "llvm.intr.log10"(%{{.*}}) : (f32) -> f32
  %3 = call float @llvm.log10.f32(float %0)
  ; CHECK:  "llvm.intr.log10"(%{{.*}}) : (vector<8xf32>) -> vector<8xf32>
  %4 = call <8 x float> @llvm.log10.v8f32(<8 x float> %1)
  ret void
}

; CHECK-LABEL:  llvm.func @log2_test
define void @log2_test(float %0, <8 x float> %1)  {
  ; CHECK:  "llvm.intr.log2"(%{{.*}}) : (f32) -> f32
  %3 = call float @llvm.log2.f32(float %0)
  ; CHECK: "llvm.intr.log2"(%{{.*}}) : (vector<8xf32>) -> vector<8xf32>
  %4 = call <8 x float> @llvm.log2.v8f32(<8 x float> %1)
  ret void
}

; CHECK-LABEL:  llvm.func @fabs_test
define void @fabs_test(float %0, <8 x float> %1) {
  ; CHECK: "llvm.intr.fabs"(%{{.*}}) : (f32) -> f32
  %3 = call float @llvm.fabs.f32(float %0)
  ; CHECK: "llvm.intr.fabs"(%{{.*}}) : (vector<8xf32>) -> vector<8xf32>
  %4 = call <8 x float> @llvm.fabs.v8f32(<8 x float> %1)
  ret void
}
; CHECK-LABEL:  llvm.func @sqrt_test
define void @sqrt_test(float %0, <8 x float> %1) {
  ; CHECK: "llvm.intr.sqrt"(%{{.*}}) : (f32) -> f32
  %3 = call float @llvm.sqrt.f32(float %0)
  ; CHECK: "llvm.intr.sqrt"(%{{.*}}) : (vector<8xf32>) -> vector<8xf32>
  %4 = call <8 x float> @llvm.sqrt.v8f32(<8 x float> %1)
  ret void
}
; CHECK-LABEL:  llvm.func @ceil_test
define void @ceil_test(float %0, <8 x float> %1) {
  ; CHECK: "llvm.intr.ceil"(%{{.*}}) : (f32) -> f32
  %3 = call float @llvm.ceil.f32(float %0)
  ; CHECK: "llvm.intr.ceil"(%{{.*}}) : (vector<8xf32>) -> vector<8xf32>
  %4 = call <8 x float> @llvm.ceil.v8f32(<8 x float> %1)
  ret void
}
; CHECK-LABEL:  llvm.func @floor_test
define void @floor_test(float %0, <8 x float> %1) {
  ; CHECK: "llvm.intr.floor"(%{{.*}}) : (f32) -> f32
  %3 = call float @llvm.floor.f32(float %0)
  ; CHECK: "llvm.intr.floor"(%{{.*}}) : (vector<8xf32>) -> vector<8xf32>
  %4 = call <8 x float> @llvm.floor.v8f32(<8 x float> %1)
  ret void
}
; CHECK-LABEL:  llvm.func @cos_test
define void @cos_test(float %0, <8 x float> %1) {
  ; CHECK: "llvm.intr.cos"(%{{.*}}) : (f32) -> f32
  %3 = call float @llvm.cos.f32(float %0)
  ; CHECK: "llvm.intr.cos"(%{{.*}}) : (vector<8xf32>) -> vector<8xf32>
  %4 = call <8 x float> @llvm.cos.v8f32(<8 x float> %1)
  ret void
}

; CHECK-LABEL:  llvm.func @copysign_test
define void @copysign_test(float %0, float %1, <8 x float> %2, <8 x float> %3) {
  ; CHECK:  "llvm.intr.copysign"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %5 = call float @llvm.copysign.f32(float %0, float %1)
  ; CHECK:  "llvm.intr.copysign"(%{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %6 = call <8 x float> @llvm.copysign.v8f32(<8 x float> %2, <8 x float> %3)
  ret void
}
; CHECK-LABEL:  llvm.func @pow_test
define void @pow_test(float %0, float %1, <8 x float> %2, <8 x float> %3) {
  ; CHECK:  "llvm.intr.pow"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %5 = call float @llvm.pow.f32(float %0, float %1)
  ; CHECK:  "llvm.intr.pow"(%{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %6 = call <8 x float> @llvm.pow.v8f32(<8 x float> %2, <8 x float> %3)
  ret void
}
; CHECK-LABEL:  llvm.func @bitreverse_test
define void @bitreverse_test(i32 %0, <8 x i32> %1) {
  ; CHECK:   "llvm.intr.bitreverse"(%{{.*}}) : (i32) -> i32
  %3 = call i32 @llvm.bitreverse.i32(i32 %0)
  ; CHECK:   "llvm.intr.bitreverse"(%{{.*}}) : (vector<8xi32>) -> vector<8xi32>
  %4 = call <8 x i32> @llvm.bitreverse.v8i32(<8 x i32> %1)
  ret void
}
; CHECK-LABEL:  llvm.func @ctlz_test
define void @ctlz_test(i32 %0, <8 x i32> %1) {
  ; CHECK-DAG:   %[[falseval1:.+]] = llvm.mlir.constant(false) : i1
  ; CHECK-DAG:   %[[falseval2:.+]] = llvm.mlir.constant(false) : i1
  ; CHECK:   "llvm.intr.ctlz"(%{{.*}}, %[[falseval2]]) : (i32, i1) -> i32
  %3 = call i32 @llvm.ctlz.i32(i32 %0, i1 false)
  ; CHECK:   "llvm.intr.ctlz"(%{{.*}}, %[[falseval1]]) : (vector<8xi32>, i1) -> vector<8xi32>
  %4 = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %1, i1 false)
  ret void
}
; CHECK-LABEL:  llvm.func @cttz_test
define void @cttz_test(i32 %0, <8 x i32> %1) {
  ; CHECK-DAG:   %[[falseval1:.+]] = llvm.mlir.constant(false) : i1
  ; CHECK-DAG:   %[[falseval2:.+]] = llvm.mlir.constant(false) : i1
  ; CHECK:   "llvm.intr.cttz"(%{{.*}}, %[[falseval2]]) : (i32, i1) -> i32
  %3 = call i32 @llvm.cttz.i32(i32 %0, i1 false)
  ; CHECK:   "llvm.intr.cttz"(%{{.*}}, %[[falseval1]]) : (vector<8xi32>, i1) -> vector<8xi32>
  %4 = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %1, i1 false)
  ret void
}

; CHECK-LABEL:  llvm.func @ctpop_test
define void @ctpop_test(i32 %0, <8 x i32> %1) {
  ; CHECK:   "llvm.intr.ctpop"(%{{.*}}) : (i32) -> i32
  %3 = call i32 @llvm.ctpop.i32(i32 %0)
  ; CHECK:   "llvm.intr.ctpop"(%{{.*}}) : (vector<8xi32>) -> vector<8xi32>
  %4 = call <8 x i32> @llvm.ctpop.v8i32(<8 x i32> %1)
  ret void
}

; CHECK-LABEL:  llvm.func @maximum_test
define void @maximum_test(float %0, float %1, <8 x float> %2, <8 x float> %3) {
  ; CHECK:   "llvm.intr.maximum"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %5 = call float @llvm.maximum.f32(float %0, float %1)
  ; CHECK:   "llvm.intr.maximum"(%{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %6 = call <8 x float> @llvm.maximum.v8f32(<8 x float> %2, <8 x float> %3)
  ret void
}

; CHECK-LABEL:    llvm.func @minimum_test
define void @minimum_test(float %0, float %1, <8 x float> %2, <8 x float> %3) {
  ; CHECK:   "llvm.intr.minimum"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %5 = call float @llvm.minimum.f32(float %0, float %1)
  ; CHECK:   "llvm.intr.minimum"(%{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %6 = call <8 x float> @llvm.minimum.v8f32(<8 x float> %2, <8 x float> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @maxnum_test
define void @maxnum_test(float %0, float %1, <8 x float> %2, <8 x float> %3) {
  ; CHECK:   "llvm.intr.maxnum"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %5 = call float @llvm.maxnum.f32(float %0, float %1)
  ; CHECK:   "llvm.intr.maxnum"(%{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %6 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %2, <8 x float> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @minnum_test
define void @minnum_test(float %0, float %1, <8 x float> %2, <8 x float> %3) {
  ; CHECK:   "llvm.intr.minnum"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %5 = call float @llvm.minnum.f32(float %0, float %1)
  ; CHECK:   "llvm.intr.minnum"(%{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %6 = call <8 x float> @llvm.minnum.v8f32(<8 x float> %2, <8 x float> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @smax_test
define void @smax_test(i32 %0, i32 %1, <8 x i32> %2, <8 x i32> %3) {
  ; CHECK:  "llvm.intr.smax"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  %5 = call i32 @llvm.smax.i32(i32 %0, i32 %1)
  ; CHECK:  "llvm.intr.smax"(%{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  %6 = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %2, <8 x i32> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @smin_test
define void @smin_test(i32 %0, i32 %1, <8 x i32> %2, <8 x i32> %3) {
  ; CHECK:   "llvm.intr.smin"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  %5 = call i32 @llvm.smin.i32(i32 %0, i32 %1)
  ; CHECK:   "llvm.intr.smin"(%{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  %6 = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %2, <8 x i32> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @umax_test
define void @umax_test(i32 %0, i32 %1, <8 x i32> %2, <8 x i32> %3) {
  ; CHECK:   "llvm.intr.umax"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  %5 = call i32 @llvm.umax.i32(i32 %0, i32 %1)
  ; CHECK:   "llvm.intr.umax"(%{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  %6 = call <8 x i32> @llvm.umax.v8i32(<8 x i32> %2, <8 x i32> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @umin_test
define void @umin_test(i32 %0, i32 %1, <8 x i32> %2, <8 x i32> %3) {
  ; CHECK:   "llvm.intr.umin"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  %5 = call i32 @llvm.umin.i32(i32 %0, i32 %1)
  ; CHECK:   "llvm.intr.umin"(%{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  %6 = call <8 x i32> @llvm.umin.v8i32(<8 x i32> %2, <8 x i32> %3)
  ret void
}
; CHECK-LABEL:  llvm.func @vector_reductions
define void @vector_reductions(float %0, <8 x float> %1, <8 x i32> %2) {
  ; CHECK: "llvm.intr.vector.reduce.add"(%{{.*}}) : (vector<8xi32>) -> i32
  %4 = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %2)
  ; CHECK: "llvm.intr.vector.reduce.and"(%{{.*}}) : (vector<8xi32>) -> i32  
  %5 = call i32 @llvm.vector.reduce.and.v8i32(<8 x i32> %2)
  ; CHECK: "llvm.intr.vector.reduce.fmax"(%{{.*}}) : (vector<8xf32>) -> f32
  %6 = call float @llvm.vector.reduce.fmax.v8f32(<8 x float> %1)
  ; CHECK: "llvm.intr.vector.reduce.fmin"(%{{.*}}) : (vector<8xf32>) -> f32
  %7 = call float @llvm.vector.reduce.fmin.v8f32(<8 x float> %1)
  ; CHECK: "llvm.intr.vector.reduce.mul"(%{{.*}}) : (vector<8xi32>) -> i32  
  %8 = call i32 @llvm.vector.reduce.mul.v8i32(<8 x i32> %2)
  ; CHECK: "llvm.intr.vector.reduce.or"(%{{.*}}) : (vector<8xi32>) -> i32
  %9 = call i32 @llvm.vector.reduce.or.v8i32(<8 x i32> %2)
  ; CHECK: "llvm.intr.vector.reduce.smax"(%{{.*}}) : (vector<8xi32>) -> i32
  %10 = call i32 @llvm.vector.reduce.smax.v8i32(<8 x i32> %2)
  ; CHECK: "llvm.intr.vector.reduce.smin"(%{{.*}}) : (vector<8xi32>) -> i32
  %11 = call i32 @llvm.vector.reduce.smin.v8i32(<8 x i32> %2)
  ; CHECK: "llvm.intr.vector.reduce.umax"(%{{.*}}) : (vector<8xi32>) -> i32
  %12 = call i32 @llvm.vector.reduce.umax.v8i32(<8 x i32> %2)
  ; CHECK: "llvm.intr.vector.reduce.umin"(%{{.*}}) : (vector<8xi32>) -> i32
  %13 = call i32 @llvm.vector.reduce.umin.v8i32(<8 x i32> %2)
  ; TODO: vector reduce fadd and fmul should be handled specially.
  %14 = call float @llvm.vector.reduce.fadd.v8f32(float %0, <8 x float> %1)
  %15 = call float @llvm.vector.reduce.fmul.v8f32(float %0, <8 x float> %1)
  %16 = call reassoc float @llvm.vector.reduce.fadd.v8f32(float %0, <8 x float> %1)
  %17 = call reassoc float @llvm.vector.reduce.fmul.v8f32(float %0, <8 x float> %1)
  ; CHECK:  "llvm.intr.vector.reduce.xor"(%{{.*}}) : (vector<8xi32>) -> i32
  %18 = call i32 @llvm.vector.reduce.xor.v8i32(<8 x i32> %2)
  ret void
}

; TODO: matrix intrinsic should be handled specially.
define void @matrix_intrinsics(<64 x float> %0, <48 x float> %1, float* %2, i64 %3) {
  %5 = call <12 x float> @llvm.matrix.multiply.v12f32.v64f32.v48f32(<64 x float> %0, <48 x float> %1, i32 4, i32 16, i32 3)
  %6 = call <48 x float> @llvm.matrix.transpose.v48f32(<48 x float> %1, i32 3, i32 16)
  %7 = call <48 x float> @llvm.matrix.column.major.load.v48f32.i64(float* align 4 %2, i64 %3, i1 false, i32 3, i32 16)
  call void @llvm.matrix.column.major.store.v48f32.i64(<48 x float> %7, float* align 4 %2, i64 %3, i1 false, i32 3, i32 16)
  ret void
}

; CHECK-LABEL: llvm.func @get_active_lane_mask
define <7 x i1> @get_active_lane_mask(i64 %0, i64 %1) {
  ; CHECK: llvm.intr.get.active.lane.mask %{{.*}}, %{{.*}} : i64, i64 to vector<7xi1>
  %3 = call <7 x i1> @llvm.get.active.lane.mask.v7i1.i64(i64 %0, i64 %1)
  ret <7 x i1> %3
}

; TODO: masked load store intrinsics should be handled specially.
define void @masked_load_store_intrinsics(<7 x float>* %0, <7 x i1> %1) {
  %3 = call <7 x float> @llvm.masked.load.v7f32.p0v7f32(<7 x float>* %0, i32 1, <7 x i1> %1, <7 x float> undef)
  %4 = call <7 x float> @llvm.masked.load.v7f32.p0v7f32(<7 x float>* %0, i32 1, <7 x i1> %1, <7 x float> %3)
  call void @llvm.masked.store.v7f32.p0v7f32(<7 x float> %4, <7 x float>* %0, i32 1, <7 x i1> %1)
  ret void
}

; TODO: masked gather scatter intrinsics should be handled specially.
define void @masked_gather_scatter_intrinsics(<7 x float*> %0, <7 x i1> %1) {
  %3 = call <7 x float> @llvm.masked.gather.v7f32.v7p0f32(<7 x float*> %0, i32 1, <7 x i1> %1, <7 x float> undef)
  %4 = call <7 x float> @llvm.masked.gather.v7f32.v7p0f32(<7 x float*> %0, i32 1, <7 x i1> %1, <7 x float> %3)
  call void @llvm.masked.scatter.v7f32.v7p0f32(<7 x float> %4, <7 x float*> %0, i32 1, <7 x i1> %1)
  ret void
}

; CHECK-LABEL:  llvm.func @masked_expand_compress_intrinsics
define void @masked_expand_compress_intrinsics(float* %0, <7 x i1> %1, <7 x float> %2) {
  ; CHECK: %[[val1:.+]] = "llvm.intr.masked.expandload"(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<f32>, vector<7xi1>, vector<7xf32>) -> vector<7xf32>
  %4 = call <7 x float> @llvm.masked.expandload.v7f32(float* %0, <7 x i1> %1, <7 x float> %2)
  ; CHECK: "llvm.intr.masked.compressstore"(%[[val1]], %{{.*}}, %{{.*}}) : (vector<7xf32>, !llvm.ptr<f32>, vector<7xi1>) -> ()
  call void @llvm.masked.compressstore.v7f32(<7 x float> %4, float* %0, <7 x i1> %1)
  ret void
}

; CHECK-LABEL:  llvm.func @memcpy_test
define void @memcpy_test(i32 %0, i8* %1, i8* %2) {
  ; CHECK: %[[falseval1:.+]] = llvm.mlir.constant(false) : i1
  ; CHECK: %[[constant:.+]] = llvm.mlir.constant(10 : i64) : i64
  ; CHECK: %[[falseval2:.+]] = llvm.mlir.constant(false) : i1
  ; CHECK: "llvm.intr.memcpy"(%{{.*}}, %{{.*}}, %{{.*}}, %[[falseval2]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i32, i1) -> ()
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %1, i8* %2, i32 %0, i1 false)
  ; CHECK: "llvm.intr.memcpy.inline"(%{{.*}}, %{{.*}}, %[[constant]], %[[falseval1]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* %1, i8* %2, i64 10, i1 false)
  ret void
}

; CHECK-LABEL:  llvm.func @memmove_test
define void @memmove_test(i32 %0, i8* %1, i8* %2) {
  ; CHECK: %[[falseval:.+]] = llvm.mlir.constant(false) : i1
  ; CHECK: "llvm.intr.memmove"(%{{.*}}, %{{.*}}, %{{.*}}, %[[falseval]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i32, i1) -> ()
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %1, i8* %2, i32 %0, i1 false)
  ret void
}

; CHECK-LABEL:  llvm.func @memset_test
define void @memset_test(i32 %0, i8* %1, i8 %2) {
  ; CHECK: %[[falseval:.+]] = llvm.mlir.constant(false) : i1
  ; CHECK: "llvm.intr.memset"(%{{.*}}, %{{.*}}, %{{.*}}, %[[falseval]]) : (!llvm.ptr<i8>, i8, i32, i1) -> ()
  call void @llvm.memset.p0i8.i32(i8* %1, i8 %2, i32 %0, i1 false)
  ret void
}

; CHECK-LABEL:  llvm.func @sadd_with_overflow_test
define void @sadd_with_overflow_test(i32 %0, i32 %1, <8 x i32> %2, <8 x i32> %3) {
  ; CHECK: "llvm.intr.sadd.with.overflow"(%{{.*}}, %{{.*}}) : (i32, i32) -> !llvm.struct<(i32, i1)>
  %5 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %0, i32 %1)
  ; CHECK: "llvm.intr.sadd.with.overflow"(%{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  %6 = call { <8 x i32>, <8 x i1> } @llvm.sadd.with.overflow.v8i32(<8 x i32> %2, <8 x i32> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @uadd_with_overflow_test
define void @uadd_with_overflow_test(i32 %0, i32 %1, <8 x i32> %2, <8 x i32> %3) {
  ; CHECK: "llvm.intr.uadd.with.overflow"(%{{.*}}, %{{.*}}) : (i32, i32) -> !llvm.struct<(i32, i1)>
  %5 = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %0, i32 %1)
  ; CHECK: "llvm.intr.uadd.with.overflow"(%{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  %6 = call { <8 x i32>, <8 x i1> } @llvm.uadd.with.overflow.v8i32(<8 x i32> %2, <8 x i32> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @ssub_with_overflow_test
define void @ssub_with_overflow_test(i32 %0, i32 %1, <8 x i32> %2, <8 x i32> %3) {
  ; CHECK: "llvm.intr.ssub.with.overflow"(%{{.*}}, %{{.*}}) : (i32, i32) -> !llvm.struct<(i32, i1)>
  %5 = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %0, i32 %1)
  ; CHECK: "llvm.intr.ssub.with.overflow"(%{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  %6 = call { <8 x i32>, <8 x i1> } @llvm.ssub.with.overflow.v8i32(<8 x i32> %2, <8 x i32> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @usub_with_overflow_test
define void @usub_with_overflow_test(i32 %0, i32 %1, <8 x i32> %2, <8 x i32> %3) {
  ; CHECK: "llvm.intr.usub.with.overflow"(%{{.*}}, %{{.*}}) : (i32, i32) -> !llvm.struct<(i32, i1)>
  %5 = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %0, i32 %1)
  ; CHECK: "llvm.intr.usub.with.overflow"(%{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  %6 = call { <8 x i32>, <8 x i1> } @llvm.usub.with.overflow.v8i32(<8 x i32> %2, <8 x i32> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @smul_with_overflow_test
define void @smul_with_overflow_test(i32 %0, i32 %1, <8 x i32> %2, <8 x i32> %3) {
  ; CHECK: "llvm.intr.smul.with.overflow"(%{{.*}}, %{{.*}}) : (i32, i32) -> !llvm.struct<(i32, i1)>
  %5 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %0, i32 %1)
  ; CHECK: "llvm.intr.smul.with.overflow"(%{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  %6 = call { <8 x i32>, <8 x i1> } @llvm.smul.with.overflow.v8i32(<8 x i32> %2, <8 x i32> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @umul_with_overflow_test
define void @umul_with_overflow_test(i32 %0, i32 %1, <8 x i32> %2, <8 x i32> %3) {
  ; CHECK: "llvm.intr.umul.with.overflow"(%{{.*}}, %{{.*}}) : (i32, i32) -> !llvm.struct<(i32, i1)>
  %5 = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %0, i32 %1)
  ; CHECK: "llvm.intr.umul.with.overflow"(%{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  %6 = call { <8 x i32>, <8 x i1> } @llvm.umul.with.overflow.v8i32(<8 x i32> %2, <8 x i32> %3)
  ret void
}

; CHECK-LABEL:  llvm.func @coro_id
define void @coro_id(i32 %0, i8* %1) {
  ; CHECK: llvm.intr.coro.id %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.token
  %3 = call token @llvm.coro.id(i32 %0, i8* %1, i8* %1, i8* null)
  ret void
}

; CHECK-LABEL:  llvm.func @coro_begin
define void @coro_begin(i32 %0, i8* %1) {
  ; CHECK: llvm.intr.coro.id %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.token
  %3 = call token @llvm.coro.id(i32 %0, i8* %1, i8* %1, i8* null)
  ; CHECK: llvm.intr.coro.begin %{{.*}}, %{{.*}} : !llvm.ptr<i8>
  %4 = call i8* @llvm.coro.begin(token %3, i8* %1)
  ret void
}

; CHECK-LABEL:  llvm.func @coro_size()
define void @coro_size() {
  ; CHECK: llvm.intr.coro.size : i64
  %1 = call i64 @llvm.coro.size.i64()
  ; CHECK: llvm.intr.coro.size : i32
  %2 = call i32 @llvm.coro.size.i32()
  ret void
}
; CHECK-LABEL:  llvm.func @coro_align()
define void @coro_align() {
  ; CHECK: llvm.intr.coro.align : i64
  %1 = call i64 @llvm.coro.align.i64()
  ; CHECK: llvm.intr.coro.align : i32
  %2 = call i32 @llvm.coro.align.i32()
  ret void
}

; CHECK-LABEL:  llvm.func @coro_save
define void @coro_save(i8* %0) {
  ; CHECK: llvm.intr.coro.save %{{.*}} : !llvm.token
  %2 = call token @llvm.coro.save(i8* %0)
  ret void
}

; CHECK-LABEL:  llvm.func @coro_suspend
define void @coro_suspend(i32 %0, i1 %1, i8* %2) {
  ; CHECK: llvm.intr.coro.id %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.token
  %4 = call token @llvm.coro.id(i32 %0, i8* %2, i8* %2, i8* null)
  ; CHECK: llvm.intr.coro.suspend %{{.*}}, %{{.*}} : i8
  %5 = call i8 @llvm.coro.suspend(token %4, i1 %1)
  ret void
}

; CHECK-LABEL:  llvm.func @coro_end
define void @coro_end(i8* %0, i1 %1) {
  ; CHECK:  llvm.intr.coro.end 
  call i1 @llvm.coro.end(i8* %0, i1 %1)
  ret void
}

; CHECK-LABEL:  llvm.func @coro_free
define void @coro_free(i32 %0, i8* %1) {
  ; CHECK: llvm.intr.coro.id %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.token
  %3 = call token @llvm.coro.id(i32 %0, i8* %1, i8* %1, i8* null)
  ; CHECK: llvm.intr.coro.free %{{.*}}, %{{.*}} : !llvm.ptr<i8>
  %4 = call i8* @llvm.coro.free(token %3, i8* %1)
  ret void
}

; CHECK-LABEL:  llvm.func @coro_resume
define void @coro_resume(i8* %0) {
  ; CHECK: llvm.intr.coro.resume %{{.*}}
  call void @llvm.coro.resume(i8* %0)
  ret void
}

; CHECK-LABEL:  llvm.func @eh_typeid_for
define void @eh_typeid_for(i8* %0) {
  ; CHECK: llvm.intr.eh.typeid.for %{{.*}} : i32
  %2 = call i32 @llvm.eh.typeid.for(i8* %0)
  ret void
}

; CHECK-LABEL:  llvm.func @stack_save() {
define void @stack_save() {
  ; CHECK: llvm.intr.stacksave : !llvm.ptr<i8>
  %1 = call i8* @llvm.stacksave()
  ret void
}

; CHECK-LABEL:  llvm.func @stack_restore
define void @stack_restore(i8* %0) {
  ; CHECK: llvm.intr.stackrestore %{{.*}}
  call void @llvm.stackrestore(i8* %0)
  ret void
}

; CHECK-LABEL:  llvm.func @vector_predication_intrinsics
define void @vector_predication_intrinsics(<8 x i32> %0, <8 x i32> %1, <8 x float> %2, <8 x float> %3, <8 x i64> %4, <8 x double> %5, <8 x i32*> %6, i32 %7, float %8, i32* %9, float* %10, <8 x i1> %11, i32 %12) {
  ; CHECK: "llvm.intr.vp.add"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %14 = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.sub"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>  
  %15 = call <8 x i32> @llvm.vp.sub.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.mul"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %16 = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.sdiv"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %17 = call <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.udiv"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %18 = call <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.srem"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %19 = call <8 x i32> @llvm.vp.srem.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.urem"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %20 = call <8 x i32> @llvm.vp.urem.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.ashr"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %21 = call <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.lshr"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %22 = call <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.shl"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %23 = call <8 x i32> @llvm.vp.shl.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.or"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %24 = call <8 x i32> @llvm.vp.or.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.and"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %25 = call <8 x i32> @llvm.vp.and.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.xor"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  %26 = call <8 x i32> @llvm.vp.xor.v8i32(<8 x i32> %0, <8 x i32> %1, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.fadd"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  %27 = call <8 x float> @llvm.vp.fadd.v8f32(<8 x float> %2, <8 x float> %3, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.fsub"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  %28 = call <8 x float> @llvm.vp.fsub.v8f32(<8 x float> %2, <8 x float> %3, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.fmul"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  %29 = call <8 x float> @llvm.vp.fmul.v8f32(<8 x float> %2, <8 x float> %3, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.fdiv"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  %30 = call <8 x float> @llvm.vp.fdiv.v8f32(<8 x float> %2, <8 x float> %3, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.frem"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  %31 = call <8 x float> @llvm.vp.frem.v8f32(<8 x float> %2, <8 x float> %3, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.fneg"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  %32 = call <8 x float> @llvm.vp.fneg.v8f32(<8 x float> %2, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.fma"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  %33 = call <8 x float> @llvm.vp.fma.v8f32(<8 x float> %2, <8 x float> %3, <8 x float> %3, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.add"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  %34 = call i32 @llvm.vp.reduce.add.v8i32(i32 %7, <8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.mul"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  %35 = call i32 @llvm.vp.reduce.mul.v8i32(i32 %7, <8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.and"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  %36 = call i32 @llvm.vp.reduce.and.v8i32(i32 %7, <8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.or"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  %37 = call i32 @llvm.vp.reduce.or.v8i32(i32 %7, <8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.xor"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  %38 = call i32 @llvm.vp.reduce.xor.v8i32(i32 %7, <8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.smax"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  %39 = call i32 @llvm.vp.reduce.smax.v8i32(i32 %7, <8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.smin"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  %40 = call i32 @llvm.vp.reduce.smin.v8i32(i32 %7, <8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.umax"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  %41 = call i32 @llvm.vp.reduce.umax.v8i32(i32 %7, <8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.umin"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  %42 = call i32 @llvm.vp.reduce.umin.v8i32(i32 %7, <8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.fadd"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  %43 = call float @llvm.vp.reduce.fadd.v8f32(float %8, <8 x float> %2, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.fmul"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  %44 = call float @llvm.vp.reduce.fmul.v8f32(float %8, <8 x float> %2, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.fmax"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  %45 = call float @llvm.vp.reduce.fmax.v8f32(float %8, <8 x float> %2, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.reduce.fmin"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  %46 = call float @llvm.vp.reduce.fmin.v8f32(float %8, <8 x float> %2, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.select"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi1>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %47 = call <8 x i32> @llvm.vp.select.v8i32(<8 x i1> %11, <8 x i32> %0, <8 x i32> %1, i32 %12)
  ; CHECK: "llvm.intr.vp.merge"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi1>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %48 = call <8 x i32> @llvm.vp.merge.v8i32(<8 x i1> %11, <8 x i32> %0, <8 x i32> %1, i32 %12)
  ; CHECK: "llvm.intr.vp.store"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, !llvm.ptr<i32>, vector<8xi1>, i32) -> ()
  call void @llvm.vp.store.v8i32.p0i32(<8 x i32> %0, i32* %9, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.load"(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i32>, vector<8xi1>, i32) -> vector<8xi32>
  %49 = call <8 x i32> @llvm.vp.load.v8i32.p0i32(i32* %9, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.experimental.vp.strided.store"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, !llvm.ptr<i32>, i32, vector<8xi1>, i32) -> ()
  call void @llvm.experimental.vp.strided.store.v8i32.p0i32.i32(<8 x i32> %0, i32* %9, i32 %7, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.experimental.vp.strided.load"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i32>, i32, vector<8xi1>, i32) -> vector<8xi32>
  %50 = call <8 x i32> @llvm.experimental.vp.strided.load.v8i32.p0i32.i32(i32* %9, i32 %7, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.trunc"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi64>, vector<8xi1>, i32) -> vector<8xi32>
  %51 = call <8 x i32> @llvm.vp.trunc.v8i32.v8i64(<8 x i64> %4, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.zext"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi1>, i32) -> vector<8xi64>
  %52 = call <8 x i64> @llvm.vp.zext.v8i64.v8i32(<8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.sext"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi32>, vector<8xi1>, i32) -> vector<8xi64>
  %53 = call <8 x i64> @llvm.vp.sext.v8i64.v8i32(<8 x i32> %0, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.fptrunc"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf64>, vector<8xi1>, i32) -> vector<8xf32>
  %54 = call <8 x float> @llvm.vp.fptrunc.v8f32.v8f64(<8 x double> %5, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.fpext"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf32>, vector<8xi1>, i32) -> vector<8xf64>
  %55 = call <8 x double> @llvm.vp.fpext.v8f64.v8f32(<8 x float> %2, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.fptoui"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf64>, vector<8xi1>, i32) -> vector<8xi64>
  %56 = call <8 x i64> @llvm.vp.fptoui.v8i64.v8f64(<8 x double> %5, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.fptosi"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xf64>, vector<8xi1>, i32) -> vector<8xi64>
  %57 = call <8 x i64> @llvm.vp.fptosi.v8i64.v8f64(<8 x double> %5, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.ptrtoint"(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.vec<8 x ptr<i32>>, vector<8xi1>, i32) -> vector<8xi64>
  %58 = call <8 x i64> @llvm.vp.ptrtoint.v8i64.v8p0i32(<8 x i32*> %6, <8 x i1> %11, i32 %12)
  ; CHECK: "llvm.intr.vp.inttoptr"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<8xi64>, vector<8xi1>, i32) -> !llvm.vec<8 x ptr<i32>>  
  %59 = call <8 x i32*> @llvm.vp.inttoptr.v8p0i32.v8i64(<8 x i64> %4, <8 x i1> %11, i32 %12)
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float)
declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>)
declare float @llvm.fma.f32(float, float, float)
declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>)
declare void @llvm.prefetch.p0i8(i8* nocapture readonly, i32 immarg, i32 immarg, i32)
declare float @llvm.exp.f32(float)
declare <8 x float> @llvm.exp.v8f32(<8 x float>)
declare float @llvm.exp2.f32(float)
declare <8 x float> @llvm.exp2.v8f32(<8 x float>)
declare float @llvm.log.f32(float)
declare <8 x float> @llvm.log.v8f32(<8 x float>)
declare float @llvm.log10.f32(float)
declare <8 x float> @llvm.log10.v8f32(<8 x float>)
declare float @llvm.log2.f32(float)
declare <8 x float> @llvm.log2.v8f32(<8 x float>)
declare float @llvm.fabs.f32(float)
declare <8 x float> @llvm.fabs.v8f32(<8 x float>)
declare float @llvm.sqrt.f32(float)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>)
declare float @llvm.ceil.f32(float)
declare <8 x float> @llvm.ceil.v8f32(<8 x float>)
declare float @llvm.floor.f32(float)
declare <8 x float> @llvm.floor.v8f32(<8 x float>)
declare float @llvm.cos.f32(float)
declare <8 x float> @llvm.cos.v8f32(<8 x float>)
declare float @llvm.copysign.f32(float, float)
declare <8 x float> @llvm.copysign.v8f32(<8 x float>, <8 x float>)
declare float @llvm.pow.f32(float, float)
declare <8 x float> @llvm.pow.v8f32(<8 x float>, <8 x float>)
declare i32 @llvm.bitreverse.i32(i32)
declare <8 x i32> @llvm.bitreverse.v8i32(<8 x i32>)
declare i32 @llvm.ctlz.i32(i32, i1 immarg)
declare <8 x i32> @llvm.ctlz.v8i32(<8 x i32>, i1 immarg)
declare i32 @llvm.cttz.i32(i32, i1 immarg)
declare <8 x i32> @llvm.cttz.v8i32(<8 x i32>, i1 immarg)
declare i32 @llvm.ctpop.i32(i32)
declare <8 x i32> @llvm.ctpop.v8i32(<8 x i32>)
declare float @llvm.maximum.f32(float, float)
declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>)
declare float @llvm.minimum.f32(float, float)
declare <8 x float> @llvm.minimum.v8f32(<8 x float>, <8 x float>)
declare float @llvm.maxnum.f32(float, float)
declare <8 x float> @llvm.maxnum.v8f32(<8 x float>, <8 x float>)
declare float @llvm.minnum.f32(float, float)
declare <8 x float> @llvm.minnum.v8f32(<8 x float>, <8 x float>)
declare i32 @llvm.smax.i32(i32, i32)
declare <8 x i32> @llvm.smax.v8i32(<8 x i32>, <8 x i32>)
declare i32 @llvm.smin.i32(i32, i32)
declare <8 x i32> @llvm.smin.v8i32(<8 x i32>, <8 x i32>)
declare i32 @llvm.umax.i32(i32, i32)
declare <8 x i32> @llvm.umax.v8i32(<8 x i32>, <8 x i32>)
declare i32 @llvm.umin.i32(i32, i32)
declare <8 x i32> @llvm.umin.v8i32(<8 x i32>, <8 x i32>)
declare i32 @llvm.vector.reduce.add.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.and.v8i32(<8 x i32>)
declare float @llvm.vector.reduce.fmax.v8f32(<8 x float>)
declare float @llvm.vector.reduce.fmin.v8f32(<8 x float>)
declare i32 @llvm.vector.reduce.mul.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.or.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.smax.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.smin.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.umax.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.umin.v8i32(<8 x i32>)
declare float @llvm.vector.reduce.fadd.v8f32(float, <8 x float>)
declare float @llvm.vector.reduce.fmul.v8f32(float, <8 x float>)
declare i32 @llvm.vector.reduce.xor.v8i32(<8 x i32>)
declare <12 x float> @llvm.matrix.multiply.v12f32.v64f32.v48f32(<64 x float>, <48 x float>, i32 immarg, i32 immarg, i32 immarg)
declare <48 x float> @llvm.matrix.transpose.v48f32(<48 x float>, i32 immarg, i32 immarg)
declare <48 x float> @llvm.matrix.column.major.load.v48f32.i64(float* nocapture, i64, i1 immarg, i32 immarg, i32 immarg)
declare void @llvm.matrix.column.major.store.v48f32.i64(<48 x float>, float* nocapture writeonly, i64, i1 immarg, i32 immarg, i32 immarg)
declare <7 x i1> @llvm.get.active.lane.mask.v7i1.i64(i64, i64)
declare <7 x float> @llvm.masked.load.v7f32.p0v7f32(<7 x float>*, i32 immarg, <7 x i1>, <7 x float>)
declare void @llvm.masked.store.v7f32.p0v7f32(<7 x float>, <7 x float>*, i32 immarg, <7 x i1>) 
declare <7 x float> @llvm.masked.gather.v7f32.v7p0f32(<7 x float*>, i32 immarg, <7 x i1>, <7 x float>)
declare void @llvm.masked.scatter.v7f32.v7p0f32(<7 x float>, <7 x float*>, i32 immarg, <7 x i1>)
declare <7 x float> @llvm.masked.expandload.v7f32(float*, <7 x i1>, <7 x float>)
declare void @llvm.masked.compressstore.v7f32(<7 x float>, float*, <7 x i1>)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64 immarg, i1 immarg)
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1 immarg)
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1 immarg)
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)
declare { <8 x i32>, <8 x i1> } @llvm.sadd.with.overflow.v8i32(<8 x i32>, <8 x i32>)
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)
declare { <8 x i32>, <8 x i1> } @llvm.uadd.with.overflow.v8i32(<8 x i32>, <8 x i32>)
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)
declare { <8 x i32>, <8 x i1> } @llvm.ssub.with.overflow.v8i32(<8 x i32>, <8 x i32>)
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32)
declare { <8 x i32>, <8 x i1> } @llvm.usub.with.overflow.v8i32(<8 x i32>, <8 x i32>)
declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32)
declare { <8 x i32>, <8 x i1> } @llvm.smul.with.overflow.v8i32(<8 x i32>, <8 x i32>)
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)
declare { <8 x i32>, <8 x i1> } @llvm.umul.with.overflow.v8i32(<8 x i32>, <8 x i32>)
declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*)
declare i8* @llvm.coro.begin(token, i8* writeonly)
declare i64 @llvm.coro.size.i64()
declare i32 @llvm.coro.size.i32()
declare i64 @llvm.coro.align.i64()
declare i32 @llvm.coro.align.i32()
declare token @llvm.coro.save(i8*)
declare i8 @llvm.coro.suspend(token, i1)
declare i1 @llvm.coro.end(i8*, i1)
declare i8* @llvm.coro.free(token, i8* nocapture readonly)
declare void @llvm.coro.resume(i8*)
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)
declare <8 x i32> @llvm.vp.add.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.sub.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.mul.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.srem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.urem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.shl.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.or.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.and.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.xor.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.fadd.v8f32(<8 x float>, <8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.fsub.v8f32(<8 x float>, <8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.fmul.v8f32(<8 x float>, <8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.fdiv.v8f32(<8 x float>, <8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.frem.v8f32(<8 x float>, <8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.fneg.v8f32(<8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.fma.v8f32(<8 x float>, <8 x float>, <8 x float>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.add.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.mul.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.and.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.or.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.xor.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.smax.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.smin.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.umax.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.umin.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare float @llvm.vp.reduce.fadd.v8f32(float, <8 x float>, <8 x i1>, i32)
declare float @llvm.vp.reduce.fmul.v8f32(float, <8 x float>, <8 x i1>, i32)
declare float @llvm.vp.reduce.fmax.v8f32(float, <8 x float>, <8 x i1>, i32)
declare float @llvm.vp.reduce.fmin.v8f32(float, <8 x float>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.select.v8i32(<8 x i1>, <8 x i32>, <8 x i32>, i32)
declare <8 x i32> @llvm.vp.merge.v8i32(<8 x i1>, <8 x i32>, <8 x i32>, i32)
declare void @llvm.vp.store.v8i32.p0i32(<8 x i32>, i32* nocapture, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.load.v8i32.p0i32(i32* nocapture, <8 x i1>, i32)
declare void @llvm.experimental.vp.strided.store.v8i32.p0i32.i32(<8 x i32>, i32* nocapture, i32, <8 x i1>, i32)
declare <8 x i32> @llvm.experimental.vp.strided.load.v8i32.p0i32.i32(i32* nocapture, i32, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.trunc.v8i32.v8i64(<8 x i64>, <8 x i1>, i32)
declare <8 x i64> @llvm.vp.zext.v8i64.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x i64> @llvm.vp.sext.v8i64.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.fptrunc.v8f32.v8f64(<8 x double>, <8 x i1>, i32)
declare <8 x double> @llvm.vp.fpext.v8f64.v8f32(<8 x float>, <8 x i1>, i32)
declare <8 x i64> @llvm.vp.fptoui.v8i64.v8f64(<8 x double>, <8 x i1>, i32)
declare <8 x i64> @llvm.vp.fptosi.v8i64.v8f64(<8 x double>, <8 x i1>, i32)
declare <8 x i64> @llvm.vp.ptrtoint.v8i64.v8p0i32(<8 x i32*>, <8 x i1>, i32)
declare <8 x i32*> @llvm.vp.inttoptr.v8p0i32.v8i64(<8 x i64>, <8 x i1>, i32)
