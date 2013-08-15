// RUN: %clang_cc1 %s -emit-llvm -o - -triple=armv7-apple-ios -std=c11 | FileCheck %s

// There isn't really anything special about iOS; it just happens to
// only deploy on processors with native atomics support, so it's a good
// way to test those code-paths.

// This work was done in pursuit of <rdar://13338582>.

// CHECK-LABEL: define arm_aapcscc void @testFloat(float*
void testFloat(_Atomic(float) *fp) {
// CHECK:      [[FP:%.*]] = alloca float*
// CHECK-NEXT: [[X:%.*]] = alloca float
// CHECK-NEXT: [[F:%.*]] = alloca float
// CHECK-NEXT: store float* {{%.*}}, float** [[FP]]

// CHECK-NEXT: [[T0:%.*]] = load float** [[FP]]
// CHECK-NEXT: store float 1.000000e+00, float* [[T0]], align 4
  __c11_atomic_init(fp, 1.0f);

// CHECK-NEXT: store float 2.000000e+00, float* [[X]], align 4
  _Atomic(float) x = 2.0f;

// CHECK-NEXT: [[T0:%.*]] = load float** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast float* [[T0]] to i32*
// CHECK-NEXT: [[T2:%.*]] = load atomic i32* [[T1]] seq_cst, align 4
// CHECK-NEXT: [[T3:%.*]] = bitcast i32 [[T2]] to float
// CHECK-NEXT: store float [[T3]], float* [[F]]
  float f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load float* [[F]], align 4
// CHECK-NEXT: [[T1:%.*]] = load float** [[FP]], align 4
// CHECK-NEXT: [[T2:%.*]] = bitcast float [[T0]] to i32
// CHECK-NEXT: [[T3:%.*]] = bitcast float* [[T1]] to i32*
// CHECK-NEXT: store atomic i32 [[T2]], i32* [[T3]] seq_cst, align 4
  *fp = f;

// CHECK-NEXT: ret void
}

// CHECK: define arm_aapcscc void @testComplexFloat([[CF:{ float, float }]]*
void testComplexFloat(_Atomic(_Complex float) *fp) {
// CHECK:      [[FP:%.*]] = alloca [[CF]]*, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[CF]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[CF]], align 4
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[CF]], align 8
// CHECK-NEXT: [[TMP1:%.*]] = alloca [[CF]], align 8
// CHECK-NEXT: store [[CF]]*

// CHECK-NEXT: [[P:%.*]] = load [[CF]]** [[FP]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[P]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]]* [[P]], i32 0, i32 1
// CHECK-NEXT: store float 1.000000e+00, float* [[T0]]
// CHECK-NEXT: store float 0.000000e+00, float* [[T1]]
  __c11_atomic_init(fp, 1.0f);

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[X]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]]* [[X]], i32 0, i32 1
// CHECK-NEXT: store float 2.000000e+00, float* [[T0]]
// CHECK-NEXT: store float 0.000000e+00, float* [[T1]]
  _Atomic(_Complex float) x = 2.0f;

// CHECK-NEXT: [[T0:%.*]] = load [[CF]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[CF]]* [[T0]] to i64*
// CHECK-NEXT: [[T2:%.*]] = load atomic i64* [[T1]] seq_cst, align 8
// CHECK-NEXT: [[T3:%.*]] = bitcast [[CF]]* [[TMP0]] to i64*
// CHECK-NEXT: store i64 [[T2]], i64* [[T3]], align 8
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[TMP0]], i32 0, i32 0
// CHECK-NEXT: [[R:%.*]] = load float* [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[TMP0]], i32 0, i32 1
// CHECK-NEXT: [[I:%.*]] = load float* [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[F]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]]* [[F]], i32 0, i32 1
// CHECK-NEXT: store float [[R]], float* [[T0]]
// CHECK-NEXT: store float [[I]], float* [[T1]]
  _Complex float f = *fp;

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[F]], i32 0, i32 0
// CHECK-NEXT: [[R:%.*]] = load float* [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[F]], i32 0, i32 1
// CHECK-NEXT: [[I:%.*]] = load float* [[T0]]
// CHECK-NEXT: [[DEST:%.*]] = load [[CF]]** [[FP]], align 4
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]]* [[TMP1]], i32 0, i32 1
// CHECK-NEXT: store float [[R]], float* [[T0]]
// CHECK-NEXT: store float [[I]], float* [[T1]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[CF]]* [[TMP1]] to i64*
// CHECK-NEXT: [[T1:%.*]] = load i64* [[T0]], align 8
// CHECK-NEXT: [[T2:%.*]] = bitcast [[CF]]* [[DEST]] to i64*
// CHECK-NEXT: store atomic i64 [[T1]], i64* [[T2]] seq_cst, align 8
  *fp = f;

// CHECK-NEXT: ret void
}

typedef struct { short x, y, z, w; } S;
// CHECK: define arm_aapcscc void @testStruct([[S:.*]]*
void testStruct(_Atomic(S) *fp) {
// CHECK:      [[FP:%.*]] = alloca [[S]]*, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[S]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[S:%.*]], align 2
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[S]], align 8
// CHECK-NEXT: store [[S]]*

// CHECK-NEXT: [[P:%.*]] = load [[S]]** [[FP]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[P]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[P]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[P]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[P]], i32 0, i32 3
// CHECK-NEXT: store i16 4, i16* [[T0]], align 2
  __c11_atomic_init(fp, (S){1,2,3,4});

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[X]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[X]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[X]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[X]], i32 0, i32 3
// CHECK-NEXT: store i16 4, i16* [[T0]], align 2
  _Atomic(S) x = (S){1,2,3,4};

// CHECK-NEXT: [[T0:%.*]] = load [[S]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[S]]* [[T0]] to i64*
// CHECK-NEXT: [[T2:%.*]] = load atomic i64* [[T1]] seq_cst, align 8
// CHECK-NEXT: [[T3:%.*]] = bitcast [[S]]* [[F]] to i64*
// CHECK-NEXT: store i64 [[T2]], i64* [[T3]], align 2
  S f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load [[S]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[S]]* [[TMP0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[S]]* [[F]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[T1]], i8* [[T2]], i32 8, i32 2, i1 false)
// CHECK-NEXT: [[T3:%.*]] = bitcast [[S]]* [[TMP0]] to i64*
// CHECK-NEXT: [[T4:%.*]] = load i64* [[T3]], align 8
// CHECK-NEXT: [[T5:%.*]] = bitcast [[S]]* [[T0]] to i64*
// CHECK-NEXT: store atomic i64 [[T4]], i64* [[T5]] seq_cst, align 8
  *fp = f;

// CHECK-NEXT: ret void
}

typedef struct { short x, y, z; } PS;
// CHECK: define arm_aapcscc void @testPromotedStruct([[APS:.*]]*
void testPromotedStruct(_Atomic(PS) *fp) {
// CHECK:      [[FP:%.*]] = alloca [[APS]]*, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[PS:%.*]], align 2
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: [[TMP1:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: store [[APS]]*

// CHECK-NEXT: [[P:%.*]] = load [[APS]]** [[FP]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[APS]]* [[P]] to i8*
// CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* [[T0]], i8 0, i64 8, i32 8, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]]* [[P]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T1]], align 2
  __c11_atomic_init(fp, (PS){1,2,3});

// CHECK-NEXT: [[T0:%.*]] = bitcast [[APS]]* [[X]] to i8*
// CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* [[T0]], i8 0, i32 8, i32 8, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]]* [[X]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T1]], align 2
  _Atomic(PS) x = (PS){1,2,3};

// CHECK-NEXT: [[T0:%.*]] = load [[APS]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[APS]]* [[T0]] to i64*
// CHECK-NEXT: [[T2:%.*]] = load atomic i64* [[T1]] seq_cst, align 8
// CHECK-NEXT: [[T3:%.*]] = bitcast [[APS]]* [[TMP0]] to i64*
// CHECK-NEXT: store i64 [[T2]], i64* [[T3]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]]* [[TMP0]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = bitcast [[PS]]* [[F]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[PS]]* [[T0]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[T1]], i8* [[T2]], i32 6, i32 2, i1 false)
  PS f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load [[APS]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[TMP1]] to i8*
// CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* [[T1]], i8 0, i32 8, i32 8, i1 false)
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[APS]]* [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[T2:%.*]] = bitcast [[PS]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = bitcast [[PS]]* [[F]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[T2]], i8* [[T3]], i32 6, i32 2, i1 false)
// CHECK-NEXT: [[T4:%.*]] = bitcast [[APS]]* [[TMP1]] to i64*
// CHECK-NEXT: [[T5:%.*]] = load i64* [[T4]], align 8
// CHECK-NEXT: [[T6:%.*]] = bitcast [[APS]]* [[T0]] to i64*
// CHECK-NEXT: store atomic i64 [[T5]], i64* [[T6]] seq_cst, align 8
  *fp = f;

// CHECK-NEXT: ret void
}

void testPromotedStructOps(_Atomic(PS) *p) {
  PS a = __c11_atomic_load(p, 5);
  __c11_atomic_store(p, a, 5);
  PS b = __c11_atomic_exchange(p, a, 5);

  _Bool v = __c11_atomic_compare_exchange_strong(p, &b, a, 5, 5);
  v = __c11_atomic_compare_exchange_weak(p, &b, a, 5, 5);
}
