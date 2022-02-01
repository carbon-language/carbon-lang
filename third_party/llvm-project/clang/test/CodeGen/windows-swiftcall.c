// RUN: %clang_cc1 -triple x86_64-unknown-windows -emit-llvm -target-cpu core2 -o - %s | FileCheck %s

#define SWIFTCALL __attribute__((swiftcall))
#define OUT __attribute__((swift_indirect_result))
#define ERROR __attribute__((swift_error_result))
#define CONTEXT __attribute__((swift_context))

// CHECK: [[STRUCT2_RESULT:@.*]] = private {{.*}} constant [[STRUCT2_TYPE:%.*]] { i32 0, i8 0, i8 undef, i8 0, i32 0, i32 0 }

/*****************************************************************************/
/****************************** PARAMETER ABIS *******************************/
/*****************************************************************************/

SWIFTCALL void indirect_result_1(OUT int *arg0, OUT float *arg1) {}
// CHECK-LABEL: define {{.*}} void @indirect_result_1(i32* noalias sret(i32*) align 4 dereferenceable(4){{.*}}, float* noalias align 4 dereferenceable(4){{.*}})

// TODO: maybe this shouldn't suppress sret.
SWIFTCALL int indirect_result_2(OUT int *arg0, OUT float *arg1) {  __builtin_unreachable(); }
// CHECK-LABEL: define {{.*}} i32 @indirect_result_2(i32* noalias align 4 dereferenceable(4){{.*}}, float* noalias align 4 dereferenceable(4){{.*}})

typedef struct { char array[1024]; } struct_reallybig;
SWIFTCALL struct_reallybig indirect_result_3(OUT int *arg0, OUT float *arg1) { __builtin_unreachable(); }
// CHECK-LABEL: define {{.*}} void @indirect_result_3({{.*}}* noalias sret({{.*}}) {{.*}}, i32* noalias align 4 dereferenceable(4){{.*}}, float* noalias align 4 dereferenceable(4){{.*}})

SWIFTCALL void context_1(CONTEXT void *self) {}
// CHECK-LABEL: define {{.*}} void @context_1(i8* swiftself

SWIFTCALL void context_2(void *arg0, CONTEXT void *self) {}
// CHECK-LABEL: define {{.*}} void @context_2(i8*{{.*}}, i8* swiftself

SWIFTCALL void context_error_1(CONTEXT int *self, ERROR float **error) {}
// CHECK-LABEL: define {{.*}} void @context_error_1(i32* swiftself{{.*}}, float** swifterror %0)
// CHECK:       [[TEMP:%.*]] = alloca float*, align 8
// CHECK:       [[T0:%.*]] = load float*, float** [[ERRORARG:%.*]], align 8
// CHECK:       store float* [[T0]], float** [[TEMP]], align 8
// CHECK:       [[T0:%.*]] = load float*, float** [[TEMP]], align 8
// CHECK:       store float* [[T0]], float** [[ERRORARG]], align 8
void test_context_error_1() {
  int x;
  float *error;
  context_error_1(&x, &error);
}
// CHECK-LABEL: define dso_local void @test_context_error_1()
// CHECK:       [[X:%.*]] = alloca i32, align 4
// CHECK:       [[ERROR:%.*]] = alloca float*, align 8
// CHECK:       [[TEMP:%.*]] = alloca swifterror float*, align 8
// CHECK:       [[T0:%.*]] = load float*, float** [[ERROR]], align 8
// CHECK:       store float* [[T0]], float** [[TEMP]], align 8
// CHECK:       call [[SWIFTCC:swiftcc]] void @context_error_1(i32* swiftself [[X]], float** swifterror [[TEMP]])
// CHECK:       [[T0:%.*]] = load float*, float** [[TEMP]], align 8
// CHECK:       store float* [[T0]], float** [[ERROR]], align 8

SWIFTCALL void context_error_2(short s, CONTEXT int *self, ERROR float **error) {}
// CHECK-LABEL: define {{.*}} void @context_error_2(i16{{.*}}, i32* swiftself{{.*}}, float** swifterror %0)

/*****************************************************************************/
/********************************** LOWERING *********************************/
/*****************************************************************************/

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double4 __attribute__((ext_vector_type(4)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int5 __attribute__((ext_vector_type(5)));
typedef int int8 __attribute__((ext_vector_type(8)));

#define TEST(TYPE)                       \
  SWIFTCALL TYPE return_##TYPE(void) {   \
    TYPE result = {};                    \
    return result;                       \
  }                                      \
  SWIFTCALL void take_##TYPE(TYPE v) {   \
  }                                      \
  void test_##TYPE() {                   \
    take_##TYPE(return_##TYPE());        \
  }

/*****************************************************************************/
/*********************************** STRUCTS *********************************/
/*****************************************************************************/

typedef struct {
} struct_empty;
TEST(struct_empty);
// CHECK-LABEL: define {{.*}} @return_struct_empty()
// CHECK:   ret void
// CHECK-LABEL: define {{.*}} @take_struct_empty()
// CHECK:   ret void

typedef struct {
  int x;
  char c0;
  char c1;
  int f0;
  int f1;
} struct_1;
TEST(struct_1);
// CHECK-LABEL: define dso_local swiftcc { i64, i64 } @return_struct_1() {{.*}}{
// CHECK:   [[RET:%.*]] = alloca [[STRUCT1:%.*]], align 4
// CHECK:   call void @llvm.memset
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT1]]* %retval to { i64, i64 }*
// CHECK:   [[GEP0:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   [[T0:%.*]] = load i64, i64* [[GEP0]], align 4
// CHECK:   [[GEP1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = load i64, i64* [[GEP1]], align 4
// CHECK:   [[R0:%.*]] = insertvalue { i64, i64 } undef, i64 [[T0]], 0
// CHECK:   [[R1:%.*]] = insertvalue { i64, i64 } [[R0]], i64 [[T1]], 1
// CHECK:   ret { i64, i64 } [[R1]]
// CHECK: }
// CHECK-LABEL: define dso_local swiftcc void @take_struct_1(i64 %0, i64 %1) {{.*}}{
// CHECK:   [[V:%.*]] = alloca [[STRUCT1:%.*]], align 4
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT1]]* [[V]] to { i64, i64 }*
// CHECK:   [[GEP0:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   store i64 %0, i64* [[GEP0]], align 4
// CHECK:   [[GEP1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   store i64 %1, i64* [[GEP1]], align 4
// CHECK:   ret void
// CHECK: }
// CHECK-LABEL: define dso_local void @test_struct_1() {{.*}}{
// CHECK:   [[AGG:%.*]] = alloca [[STRUCT1:%.*]], align 4
// CHECK:   [[RET:%.*]] = call swiftcc { i64, i64 } @return_struct_1()
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT1]]* [[AGG]] to { i64, i64 }*
// CHECK:   [[GEP0:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   [[E0:%.*]] = extractvalue { i64, i64 } [[RET]], 0
// CHECK:   store i64 [[E0]], i64* [[GEP0]], align 4
// CHECK:   [[GEP1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   [[E1:%.*]] = extractvalue { i64, i64 } [[RET]], 1
// CHECK:   store i64 [[E1]], i64* [[GEP1]], align 4
// CHECK:   [[CAST2:%.*]] = bitcast [[STRUCT1]]* [[AGG]] to { i64, i64 }*
// CHECK:   [[GEP2:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST2]], i32 0, i32 0
// CHECK:   [[V0:%.*]] = load i64, i64* [[GEP2]], align 4
// CHECK:   [[GEP3:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST2]], i32 0, i32 1
// CHECK:   [[V1:%.*]] = load i64, i64* [[GEP3]], align 4
// CHECK:   call swiftcc void @take_struct_1(i64 [[V0]], i64 [[V1]])
// CHECK:   ret void
// CHECK: }

typedef struct {
  int x;
  char c0;
  __attribute__((aligned(2))) char c1;
  int f0;
  int f1;
} struct_2;
TEST(struct_2);
// CHECK-LABEL: define dso_local swiftcc { i64, i64 } @return_struct_2() {{.*}}{
// CHECK:   [[RET:%.*]] = alloca [[STRUCT2_TYPE]], align 4
// CHECK:   [[CASTVAR:%.*]] = bitcast {{.*}} [[RET]]
// CHECK:   call void @llvm.memcpy{{.*}}({{.*}}[[CASTVAR]], {{.*}}[[STRUCT2_RESULT]]
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT2_TYPE]]* [[RET]] to { i64, i64 }*
// CHECK:   [[GEP0:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   [[T0:%.*]] = load i64, i64* [[GEP0]], align 4
// CHECK:   [[GEP1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = load i64, i64* [[GEP1]], align 4
// CHECK:   [[R0:%.*]] = insertvalue { i64, i64 } undef, i64 [[T0]], 0
// CHECK:   [[R1:%.*]] = insertvalue { i64, i64 } [[R0]], i64 [[T1]], 1
// CHECK:   ret { i64, i64 } [[R1]]
// CHECK: }
// CHECK-LABEL: define dso_local swiftcc void @take_struct_2(i64 %0, i64 %1) {{.*}}{
// CHECK:   [[V:%.*]] = alloca [[STRUCT:%.*]], align 4
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT]]* [[V]] to { i64, i64 }*
// CHECK:   [[GEP0:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   store i64 %0, i64* [[GEP0]], align 4
// CHECK:   [[GEP1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   store i64 %1, i64* [[GEP1]], align 4
// CHECK:   ret void
// CHECK: }
// CHECK-LABEL: define dso_local void @test_struct_2() {{.*}} {
// CHECK:   [[TMP:%.*]] = alloca [[STRUCT2_TYPE]], align 4
// CHECK:   [[CALL:%.*]] = call swiftcc { i64, i64 } @return_struct_2()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[STRUCT2_TYPE]]* [[TMP]] to { i64, i64 }*
// CHECK:   [[GEP:%.*]] = getelementptr inbounds {{.*}} [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T0:%.*]] = extractvalue { i64, i64 } [[CALL]], 0
// CHECK:   store i64 [[T0]], i64* [[GEP]], align 4
// CHECK:   [[GEP:%.*]] = getelementptr inbounds {{.*}} [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T0:%.*]] = extractvalue { i64, i64 } [[CALL]], 1
// CHECK:   store i64 [[T0]], i64* [[GEP]], align 4
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT2_TYPE]]* [[TMP]] to { i64, i64 }*
// CHECK:   [[GEP:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   [[R0:%.*]] = load i64, i64* [[GEP]], align 4
// CHECK:   [[GEP:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   [[R1:%.*]] = load i64, i64* [[GEP]], align 4
// CHECK:   call swiftcc void @take_struct_2(i64 [[R0]], i64 [[R1]])
// CHECK:   ret void
// CHECK: }

// There's no way to put a field randomly in the middle of an otherwise
// empty storage unit in C, so that case has to be tested in C++, which
// can use empty structs to introduce arbitrary padding.  (In C, they end up
// with size 0 and so don't affect layout.)

// Misaligned data rule.
typedef struct {
  char c0;
  __attribute__((packed)) float f;
} struct_misaligned_1;
TEST(struct_misaligned_1)
// CHECK-LABEL: define dso_local swiftcc i64 @return_struct_misaligned_1()
// CHECK:  [[RET:%.*]] = alloca [[STRUCT:%.*]], align 1
// CHECK:  [[CAST:%.*]] = bitcast [[STRUCT]]* [[RET]] to i8*
// CHECK:  call void @llvm.memset{{.*}}(i8* align 1 [[CAST]], i8 0, i64 5
// CHECK:  [[CAST:%.*]] = bitcast [[STRUCT]]* [[RET]] to { i64 }*
// CHECK:  [[GEP:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[CAST]], i32 0, i32 0
// CHECK:  [[R0:%.*]] = load i64, i64* [[GEP]], align 1
// CHECK:  ret i64 [[R0]]
// CHECK:}
// CHECK-LABEL: define dso_local swiftcc void @take_struct_misaligned_1(i64 %0) {{.*}}{
// CHECK:   [[V:%.*]] = alloca [[STRUCT:%.*]], align 1
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT]]* [[V]] to { i64 }*
// CHECK:   [[GEP:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[CAST]], i32 0, i32 0
// CHECK:   store i64 %0, i64* [[GEP]], align 1
// CHECK:   ret void
// CHECK: }
// CHECK: define dso_local void @test_struct_misaligned_1() {{.*}}{
// CHECK:   [[AGG:%.*]] = alloca [[STRUCT:%.*]], align 1
// CHECK:   [[CALL:%.*]] = call swiftcc i64 @return_struct_misaligned_1()
// CHECK:   [[T0:%.*]] = bitcast [[STRUCT]]* [[AGG]] to { i64 }*
// CHECK:   [[T1:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[T0]], i32 0, i32 0
// CHECK:   store i64 [[CALL]], i64* [[T1]], align 1
// CHECK:   [[T0:%.*]] = bitcast [[STRUCT]]* [[AGG]] to { i64 }*
// CHECK:   [[T1:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[T0]], i32 0, i32 0
// CHECK:   [[P:%.*]] = load i64, i64* [[T1]], align 1
// CHECK:   call swiftcc void @take_struct_misaligned_1(i64 [[P]])
// CHECK:   ret void
// CHECK: }

// Too many scalars.
typedef struct {
  long long x[5];
} struct_big_1;
TEST(struct_big_1)

// CHECK-LABEL: define {{.*}} void @return_struct_big_1({{.*}} noalias sret

// Should not be byval.
// CHECK-LABEL: define {{.*}} void @take_struct_big_1({{.*}}*{{( %.*)?}})

/*****************************************************************************/
/********************************* TYPE MERGING ******************************/
/*****************************************************************************/

typedef union {
  float f;
  double d;
} union_het_fp;
TEST(union_het_fp)
// CHECK-LABEL: define dso_local swiftcc i64 @return_union_het_fp()
// CHECK:  [[RET:%.*]] = alloca [[UNION:%.*]], align 8
// CHECK:  [[CAST:%.*]] = bitcast [[UNION]]* [[RET]] to i8*
// CHECK:  call void @llvm.memcpy{{.*}}(i8* align {{[0-9]+}} [[CAST]]
// CHECK:  [[CAST:%.*]] = bitcast [[UNION]]* [[RET]] to { i64 }*
// CHECK:  [[GEP:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[CAST]], i32 0, i32 0
// CHECK:  [[R0:%.*]] = load i64, i64* [[GEP]], align 8
// CHECK:  ret i64 [[R0]]
// CHECK-LABEL: define dso_local swiftcc void @take_union_het_fp(i64 %0) {{.*}}{
// CHECK:   [[V:%.*]] = alloca [[UNION:%.*]], align 8
// CHECK:   [[CAST:%.*]] = bitcast [[UNION]]* [[V]] to { i64 }*
// CHECK:   [[GEP:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[CAST]], i32 0, i32 0
// CHECK:   store i64 %0, i64* [[GEP]], align 8
// CHECK:   ret void
// CHECK: }
// CHECK-LABEL: define dso_local void @test_union_het_fp() {{.*}}{
// CHECK:   [[AGG:%.*]] = alloca [[UNION:%.*]], align 8
// CHECK:   [[CALL:%.*]] = call swiftcc i64 @return_union_het_fp()
// CHECK:   [[T0:%.*]] = bitcast [[UNION]]* [[AGG]] to { i64 }*
// CHECK:   [[T1:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[T0]], i32 0, i32 0
// CHECK:   store i64 [[CALL]], i64* [[T1]], align 8
// CHECK:   [[T0:%.*]] = bitcast [[UNION]]* [[AGG]] to { i64 }*
// CHECK:   [[T1:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[T0]], i32 0, i32 0
// CHECK:   [[V0:%.*]] = load i64, i64* [[T1]], align 8
// CHECK:   call swiftcc void @take_union_het_fp(i64 [[V0]])
// CHECK:   ret void
// CHECK: }


typedef union {
  float f1;
  float f2;
} union_hom_fp;
TEST(union_hom_fp)
// CHECK-LABEL: define dso_local void @test_union_hom_fp()
// CHECK:   [[TMP:%.*]] = alloca [[REC:%.*]], align 4
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] float @return_union_hom_fp()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG:{ float }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store float [[CALL]], float* [[T0]], align 4
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load float, float* [[T0]], align 4
// CHECK:   call [[SWIFTCC]] void @take_union_hom_fp(float [[FIRST]])
// CHECK:   ret void

typedef union {
  float f1;
  float4 fv2;
} union_hom_fp_partial;
TEST(union_hom_fp_partial)
// CHECK: define dso_local void @test_union_hom_fp_partial()
// CHECK:   [[AGG:%.*]] = alloca [[UNION:%.*]], align 16
// CHECK:   [[CALL:%.*]] = call swiftcc { float, float, float, float } @return_union_hom_fp_partial()
// CHECK:   [[CAST:%.*]] = bitcast [[UNION]]* [[AGG]] to { float, float, float, float }*
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue { float, float, float, float } [[CALL]], 0
// CHECK:   store float [[T1]], float* [[T0]], align 16
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue { float, float, float, float } [[CALL]], 1
// CHECK:   store float [[T1]], float* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 2
// CHECK:   [[T1:%.*]] = extractvalue { float, float, float, float } [[CALL]], 2
// CHECK:   store float [[T1]], float* [[T0]], align 8
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 3
// CHECK:   [[T1:%.*]] = extractvalue { float, float, float, float } [[CALL]], 3
// CHECK:   store float [[T1]], float* [[T0]], align 4
// CHECK:   [[CAST:%.*]] = bitcast [[UNION]]* [[AGG]] to { float, float, float, float }*
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 0
// CHECK:   [[V0:%.*]] = load float, float* [[T0]], align 16
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 1
// CHECK:   [[V1:%.*]] = load float, float* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 2
// CHECK:   [[V2:%.*]] = load float, float* [[T0]], align 8
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 3
// CHECK:   [[V3:%.*]] = load float, float* [[T0]], align 4
// CHECK:   call swiftcc void @take_union_hom_fp_partial(float [[V0]], float [[V1]], float [[V2]], float [[V3]])
// CHECK:   ret void
// CHECK: }

typedef union {
  struct { int x, y; } f1;
  float4 fv2;
} union_het_fpv_partial;
TEST(union_het_fpv_partial)
// CHECK-LABEL: define dso_local void @test_union_het_fpv_partial()
// CHECK:   [[AGG:%.*]] = alloca [[UNION:%.*]], align 16
// CHECK:   [[CALL:%.*]] = call swiftcc { i64, float, float } @return_union_het_fpv_partial()
// CHECK:   [[CAST:%.*]] = bitcast [[UNION]]* [[AGG]] to { i64, float, float }*
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue { i64, float, float } [[CALL]], 0
// CHECK:   store i64 [[T1]], i64* [[T0]], align 16
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue { i64, float, float } [[CALL]], 1
// CHECK:   store float [[T1]], float* [[T0]], align 8
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 2
// CHECK:   [[T1:%.*]] = extractvalue { i64, float, float } [[CALL]], 2
// CHECK:   store float [[T1]], float* [[T0]], align 4
// CHECK:   [[CAST:%.*]] = bitcast [[UNION]]* [[AGG]] to { i64, float, float }*
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 0
// CHECK:   [[V0:%.*]] = load i64, i64* [[T0]], align 16
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 1
// CHECK:   [[V1:%.*]] = load float, float* [[T0]], align 8
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 2
// CHECK:   [[V2:%.*]] = load float, float* [[T0]], align 4
// CHECK:   call swiftcc void @take_union_het_fpv_partial(i64 [[V0]], float [[V1]], float [[V2]])
// CHECK:   ret void
// CHECK: }

/*****************************************************************************/
/****************************** VECTOR LEGALIZATION **************************/
/*****************************************************************************/

TEST(int4)
// CHECK-LABEL: define {{.*}} <4 x i32> @return_int4()
// CHECK-LABEL: define {{.*}} @take_int4(<4 x i32>

TEST(int8)
// CHECK-LABEL: define {{.*}} @return_int8()
// CHECK:   [[RET:%.*]] = alloca [[REC:<8 x i32>]], align 32
// CHECK:   [[VAR:%.*]] = alloca [[REC]], align
// CHECK:   store
// CHECK:   load
// CHECK:   store
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[RET]] to [[AGG:{ <4 x i32>, <4 x i32> }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = insertvalue [[UAGG:{ <4 x i32>, <4 x i32> }]] undef, <4 x i32> [[FIRST]], 0
// CHECK:   [[T1:%.*]] = insertvalue [[UAGG]] [[T0]], <4 x i32> [[SECOND]], 1
// CHECK:   ret [[UAGG]] [[T1]]
// CHECK-LABEL: define {{.*}} @take_int8(<4 x i32> %0, <4 x i32> %1)
// CHECK:   [[V:%.*]] = alloca [[REC]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[V]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store <4 x i32> %0, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   store <4 x i32> %1, <4 x i32>* [[T0]], align
// CHECK:   ret void
// CHECK-LABEL: define dso_local void @test_int8()
// CHECK:   [[TMP1:%.*]] = alloca [[REC]], align
// CHECK:   [[TMP2:%.*]] = alloca [[REC]], align
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] [[UAGG]] @return_int8()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP1]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 0
// CHECK:   store <4 x i32> [[T1]], <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 1
// CHECK:   store <4 x i32> [[T1]], <4 x i32>* [[T0]], align
// CHECK:   [[V:%.*]] = load [[REC]], [[REC]]* [[TMP1]], align
// CHECK:   store [[REC]] [[V]], [[REC]]* [[TMP2]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP2]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   call [[SWIFTCC]] void @take_int8(<4 x i32> [[FIRST]], <4 x i32> [[SECOND]])
// CHECK:   ret void

TEST(int5)
// CHECK-LABEL: define {{.*}} @return_int5()
// CHECK:   [[RET:%.*]] = alloca [[REC:<5 x i32>]], align 32
// CHECK:   [[VAR:%.*]] = alloca [[REC]], align
// CHECK:   store
// CHECK:   load
// CHECK:   store
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[RET]] to [[AGG:{ <4 x i32>, i32 }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i32, i32* [[T0]], align
// CHECK:   [[T0:%.*]] = insertvalue [[UAGG:{ <4 x i32>, i32 }]] undef, <4 x i32> [[FIRST]], 0
// CHECK:   [[T1:%.*]] = insertvalue [[UAGG]] [[T0]], i32 [[SECOND]], 1
// CHECK:   ret [[UAGG]] [[T1]]
// CHECK-LABEL: define {{.*}} @take_int5(<4 x i32> %0, i32 %1)
// CHECK:   [[V:%.*]] = alloca [[REC]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[V]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store <4 x i32> %0, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   store i32 %1, i32* [[T0]], align
// CHECK:   ret void
// CHECK-LABEL: define dso_local void @test_int5()
// CHECK:   [[TMP1:%.*]] = alloca [[REC]], align
// CHECK:   [[TMP2:%.*]] = alloca [[REC]], align
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] [[UAGG]] @return_int5()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP1]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 0
// CHECK:   store <4 x i32> [[T1]], <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 1
// CHECK:   store i32 [[T1]], i32* [[T0]], align
// CHECK:   [[V:%.*]] = load [[REC]], [[REC]]* [[TMP1]], align
// CHECK:   store [[REC]] [[V]], [[REC]]* [[TMP2]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP2]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i32, i32* [[T0]], align
// CHECK:   call [[SWIFTCC]] void @take_int5(<4 x i32> [[FIRST]], i32 [[SECOND]])
// CHECK:   ret void

typedef struct {
  int x;
  int3 v __attribute__((packed));
} misaligned_int3;
TEST(misaligned_int3)
// CHECK-LABEL: define dso_local swiftcc void @take_misaligned_int3(i64 %0, i64 %1)
