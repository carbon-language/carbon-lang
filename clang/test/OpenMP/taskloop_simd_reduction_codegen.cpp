// RUN: %clang_cc1 -fopenmp -x c++ %s -verify -debug-info-kind=limited -emit-llvm -o - -triple powerpc64le-unknown-linux-gnu | FileCheck %s
// expected-no-diagnostics

struct S {
  float a;
  S() : a(0.0f) {}
  ~S() {}
};

#pragma omp declare reduction(+:S:omp_out.a += omp_in.a) initializer(omp_priv = omp_orig)

float g;

int a;
#pragma omp threadprivate(a)
int main (int argc, char *argv[])
{
int   i, n;
float a[100], b[100], sum, e[argc + 100];
S c[100];
float &d = g;

/* Some initializations */
n = 100;
for (i=0; i < n; i++)
  a[i] = b[i] = i * 1.0;
sum = 0.0;

#pragma omp taskloop simd reduction(+:sum, c[:n], d, e)
  for (i=0; i < n; i++) {
    sum = sum + (a[i] * b[i]);
    c[i].a = i*i;
    d += i*i;
    e[i] = i;
  }

}

// CHECK-LABEL: @main(
// CHECK:    [[RETVAL:%.*]] = alloca i32, align 4
// CHECK:    [[ARGC_ADDR:%.*]] = alloca i32, align 4
// CHECK:    [[ARGV_ADDR:%.*]] = alloca i8**, align 8
// CHECK:    [[I:%.*]] = alloca i32, align 4
// CHECK:    [[N:%.*]] = alloca i32, align 4
// CHECK:    [[A:%.*]] = alloca [100 x float], align 4
// CHECK:    [[B:%.*]] = alloca [100 x float], align 4
// CHECK:    [[SUM:%.*]] = alloca float, align 4
// CHECK:    [[SAVED_STACK:%.*]] = alloca i8*, align 8
// CHECK:    [[C:%.*]] = alloca [100 x %struct.S], align 4
// CHECK:    [[D:%.*]] = alloca float*, align 8
// CHECK:    [[AGG_CAPTURED:%.*]] = alloca [[STRUCT_ANON:%.*]], align 8
// CHECK:    [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%ident_t*
// CHECK:    [[DOTRD_INPUT_:%.*]] = alloca [4 x %struct.kmp_task_red_input_t], align 8
// CHECK:    [[DOTCAPTURE_EXPR_:%.*]] = alloca i32, align 4
// CHECK:    [[DOTCAPTURE_EXPR_9:%.*]] = alloca i32, align 4
// CHECK:    store i32 0, i32* [[RETVAL]], align 4
// CHECK:    store i32 [[ARGC:%.*]], i32* [[ARGC_ADDR]], align 4
// CHECK:    store i8** [[ARGV:%.*]], i8*** [[ARGV_ADDR]], align 8
// CHECK:    [[TMP1:%.*]] = load i32, i32* [[ARGC_ADDR]], align 4
// CHECK:    [[ADD:%.*]] = add nsw i32 [[TMP1]], 100
// CHECK:    [[TMP2:%.*]] = zext i32 [[ADD]] to i64
// CHECK:    [[VLA:%.+]] = alloca float, i64 %

// CHECK:    call void @__kmpc_taskgroup(%ident_t*
// CHECK:    [[DOTRD_INPUT_GEP_:%.*]] = getelementptr inbounds [4 x %struct.kmp_task_red_input_t], [4 x %struct.kmp_task_red_input_t]* [[DOTRD_INPUT_]], i64 0, i64 0
// CHECK:    [[TMP20:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T:%.*]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 0
// CHECK:    [[TMP21:%.*]] = bitcast float* [[SUM]] to i8*
// CHECK:    store i8* [[TMP21]], i8** [[TMP20]], align 8
// CHECK:    [[TMP22:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 1
// CHECK:    store i64 4, i64* [[TMP22]], align 8
// CHECK:    [[TMP23:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 2
// CHECK:    store i8* bitcast (void (i8*)* [[RED_INIT1:@.+]] to i8*), i8** [[TMP23]], align 8
// CHECK:    [[TMP24:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 3
// CHECK:    store i8* null, i8** [[TMP24]], align 8
// CHECK:    [[TMP25:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 4
// CHECK:    store i8* bitcast (void (i8*, i8*)* [[RED_COMB1:@.+]] to i8*), i8** [[TMP25]], align 8
// CHECK:    [[TMP26:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 5
// CHECK:    [[TMP27:%.*]] = bitcast i32* [[TMP26]] to i8*
// CHECK:    call void @llvm.memset.p0i8.i64(i8* [[TMP27]], i8 0, i64 4, i32 8, i1 false)
// CHECK:    [[DOTRD_INPUT_GEP_4:%.*]] = getelementptr inbounds [4 x %struct.kmp_task_red_input_t], [4 x %struct.kmp_task_red_input_t]* [[DOTRD_INPUT_]], i64 0, i64 1
// CHECK:    [[TMP28:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 0
// CHECK:    [[ARRAYIDX5:%.*]] = getelementptr inbounds [100 x %struct.S], [100 x %struct.S]* [[C]], i64 0, i64 0
// CHECK:    [[LB_ADD_LEN:%.*]] = add nsw i64 -1, %
// CHECK:    [[ARRAYIDX6:%.*]] = getelementptr inbounds [100 x %struct.S], [100 x %struct.S]* [[C]], i64 0, i64 [[LB_ADD_LEN]]
// CHECK:    [[TMP31:%.*]] = bitcast %struct.S* [[ARRAYIDX5]] to i8*
// CHECK:    store i8* [[TMP31]], i8** [[TMP28]], align 8
// CHECK:    [[TMP32:%.*]] = ptrtoint %struct.S* [[ARRAYIDX6]] to i64
// CHECK:    [[TMP33:%.*]] = ptrtoint %struct.S* [[ARRAYIDX5]] to i64
// CHECK:    [[TMP34:%.*]] = sub i64 [[TMP32]], [[TMP33]]
// CHECK:    [[TMP35:%.*]] = sdiv exact i64 [[TMP34]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK:    [[TMP36:%.*]] = add nuw i64 [[TMP35]], 1
// CHECK:    [[TMP37:%.*]] = mul nuw i64 [[TMP36]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK:    [[TMP38:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 1
// CHECK:    store i64 [[TMP37]], i64* [[TMP38]], align 8
// CHECK:    [[TMP39:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 2
// CHECK:    store i8* bitcast (void (i8*)* [[RED_INIT2:@.+]] to i8*), i8** [[TMP39]], align 8
// CHECK:    [[TMP40:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 3
// CHECK:    store i8* bitcast (void (i8*)* [[RED_FINI2:@.+]] to i8*), i8** [[TMP40]], align 8
// CHECK:    [[TMP41:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 4
// CHECK:    store i8* bitcast (void (i8*, i8*)* [[RED_COMB2:@.+]] to i8*), i8** [[TMP41]], align 8
// CHECK:    [[TMP42:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 5
// CHECK:    store i32 1, i32* [[TMP42]], align 8
// CHECK:    [[DOTRD_INPUT_GEP_7:%.*]] = getelementptr inbounds [4 x %struct.kmp_task_red_input_t], [4 x %struct.kmp_task_red_input_t]* [[DOTRD_INPUT_]], i64 0, i64 2
// CHECK:    [[TMP43:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 0
// CHECK:    [[TMP44:%.*]] = load float*, float** [[D]], align 8
// CHECK:    [[TMP45:%.*]] = bitcast float* [[TMP44]] to i8*
// CHECK:    store i8* [[TMP45]], i8** [[TMP43]], align 8
// CHECK:    [[TMP46:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 1
// CHECK:    store i64 4, i64* [[TMP46]], align 8
// CHECK:    [[TMP47:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 2
// CHECK:    store i8* bitcast (void (i8*)* [[RED_INIT3:@.+]] to i8*), i8** [[TMP47]], align 8
// CHECK:    [[TMP48:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 3
// CHECK:    store i8* null, i8** [[TMP48]], align 8
// CHECK:    [[TMP49:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 4
// CHECK:    store i8* bitcast (void (i8*, i8*)* [[RED_COMB3:@.+]] to i8*), i8** [[TMP49]], align 8
// CHECK:    [[TMP50:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 5
// CHECK:    [[TMP51:%.*]] = bitcast i32* [[TMP50]] to i8*
// CHECK:    call void @llvm.memset.p0i8.i64(i8* [[TMP51]], i8 0, i64 4, i32 8, i1 false)
// CHECK:    [[DOTRD_INPUT_GEP_8:%.*]] = getelementptr inbounds [4 x %struct.kmp_task_red_input_t], [4 x %struct.kmp_task_red_input_t]* [[DOTRD_INPUT_]], i64 0, i64 3
// CHECK:    [[TMP52:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 0
// CHECK:    [[TMP53:%.*]] = bitcast float* [[VLA]] to i8*
// CHECK:    store i8* [[TMP53]], i8** [[TMP52]], align 8
// CHECK:    [[TMP54:%.*]] = mul nuw i64 [[TMP2]], 4
// CHECK:    [[TMP55:%.*]] = udiv exact i64 [[TMP54]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK:    [[TMP56:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 1
// CHECK:    store i64 [[TMP54]], i64* [[TMP56]], align 8
// CHECK:    [[TMP57:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 2
// CHECK:    store i8* bitcast (void (i8*)* [[RED_INIT4:@.+]] to i8*), i8** [[TMP57]], align 8
// CHECK:    [[TMP58:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 3
// CHECK:    store i8* null, i8** [[TMP58]], align 8
// CHECK:    [[TMP59:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 4
// CHECK:    store i8* bitcast (void (i8*, i8*)* [[RED_COMB4:@.+]] to i8*), i8** [[TMP59]], align 8
// CHECK:    [[TMP60:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_RED_INPUT_T]], %struct.kmp_task_red_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 5
// CHECK:    store i32 1, i32* [[TMP60]], align 8
// CHECK:    [[TMP61:%.*]] = bitcast [4 x %struct.kmp_task_red_input_t]* [[DOTRD_INPUT_]] to i8*
// CHECK:    [[TMP62:%.*]] = call i8* @__kmpc_task_reduction_init(i32 [[TMP0]], i32 4, i8* [[TMP61]])
// CHECK:    [[TMP63:%.*]] = load i32, i32* [[N]], align 4
// CHECK:    store i32 [[TMP63]], i32* [[DOTCAPTURE_EXPR_]], align 4
// CHECK:    [[TMP64:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// CHECK:    [[SUB:%.*]] = sub nsw i32 [[TMP64]], 0
// CHECK:    [[SUB10:%.*]] = sub nsw i32 [[SUB]], 1
// CHECK:    [[ADD11:%.*]] = add nsw i32 [[SUB10]], 1
// CHECK:    [[DIV:%.*]] = sdiv i32 [[ADD11]], 1
// CHECK:    [[SUB12:%.*]] = sub nsw i32 [[DIV]], 1
// CHECK:    store i32 [[SUB12]], i32* [[DOTCAPTURE_EXPR_9]], align 4
// CHECK:    [[TMP65:%.*]] = call i8* @__kmpc_omp_task_alloc(%ident_t* %{{.+}}, i32 [[TMP0]], i32 1, i64 888, i64 72, i32 (i32, i8*)* bitcast (i32 (i32, %struct.kmp_task_t_with_privates*)* @{{.+}} to i32 (i32, i8*)*))
// CHECK:    [[TMP66:%.*]] = bitcast i8* [[TMP65]] to %struct.kmp_task_t_with_privates*
// CHECK:    [[TMP67:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T_WITH_PRIVATES:%.*]], %struct.kmp_task_t_with_privates* [[TMP66]], i32 0, i32 0
// CHECK:    [[TMP68:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T:%.*]], %struct.kmp_task_t* [[TMP67]], i32 0, i32 0
// CHECK:    [[TMP69:%.*]] = load i8*, i8** [[TMP68]], align 8
// CHECK:    [[TMP70:%.*]] = bitcast %struct.anon* [[AGG_CAPTURED]] to i8*
// CHECK:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP69]], i8* [[TMP70]], i64 72, i32 8, i1 false)
// CHECK:    [[TMP71:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T_WITH_PRIVATES]], %struct.kmp_task_t_with_privates* [[TMP66]], i32 0, i32 1
// CHECK:    [[TMP72:%.*]] = bitcast i8* [[TMP69]] to %struct.anon*
// CHECK:    [[TMP73:%.*]] = getelementptr inbounds [[STRUCT__KMP_PRIVATES_T:%.*]], %struct..kmp_privates.t* [[TMP71]], i32 0, i32 0
// CHECK:    [[TMP74:%.*]] = getelementptr inbounds [[STRUCT_ANON]], %struct.anon* [[TMP72]], i32 0, i32 1
// CHECK:    [[REF:%.*]] = load i32*, i32** [[TMP74]], align 8
// CHECK:    [[TMP75:%.*]] = load i32, i32* [[REF]], align 4
// CHECK:    store i32 [[TMP75]], i32* [[TMP73]], align 8
// CHECK:    [[TMP76:%.*]] = getelementptr inbounds [[STRUCT__KMP_PRIVATES_T]], %struct..kmp_privates.t* [[TMP71]], i32 0, i32 1
// CHECK:    [[TMP77:%.*]] = getelementptr inbounds [[STRUCT_ANON]], %struct.anon* [[TMP72]], i32 0, i32 3
// CHECK:    [[REF13:%.*]] = load [100 x float]*, [100 x float]** [[TMP77]], align 8
// CHECK:    [[TMP78:%.*]] = bitcast [100 x float]* [[TMP76]] to i8*
// CHECK:    [[TMP79:%.*]] = bitcast [100 x float]* [[REF13]] to i8*
// CHECK:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP78]], i8* [[TMP79]], i64 400, i32 4, i1 false)
// CHECK:    [[TMP80:%.*]] = getelementptr inbounds [[STRUCT__KMP_PRIVATES_T]], %struct..kmp_privates.t* [[TMP71]], i32 0, i32 2
// CHECK:    [[TMP81:%.*]] = getelementptr inbounds [[STRUCT_ANON]], %struct.anon* [[TMP72]], i32 0, i32 4
// CHECK:    [[REF14:%.*]] = load [100 x float]*, [100 x float]** [[TMP81]], align 8
// CHECK:    [[TMP82:%.*]] = bitcast [100 x float]* [[TMP80]] to i8*
// CHECK:    [[TMP83:%.*]] = bitcast [100 x float]* [[REF14]] to i8*
// CHECK:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP82]], i8* [[TMP83]], i64 400, i32 4, i1 false)
// CHECK:    [[TMP84:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T]], %struct.kmp_task_t* [[TMP67]], i32 0, i32 5
// CHECK:    store i64 0, i64* [[TMP84]], align 8
// CHECK:    [[TMP85:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T]], %struct.kmp_task_t* [[TMP67]], i32 0, i32 6
// CHECK:    [[TMP86:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_9]], align 4
// CHECK:    [[CONV15:%.*]] = sext i32 [[TMP86]] to i64
// CHECK:    store i64 [[CONV15]], i64* [[TMP85]], align 8
// CHECK:    [[TMP87:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T]], %struct.kmp_task_t* [[TMP67]], i32 0, i32 7
// CHECK:    store i64 1, i64* [[TMP87]], align 8
// CHECK:    [[TMP88:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T]], %struct.kmp_task_t* [[TMP67]], i32 0, i32 9
// CHECK:    store i8* [[TMP62]], i8** [[TMP88]], align 8
// CHECK:    [[TMP89:%.*]] = load i64, i64* [[TMP87]], align 8
// CHECK:    call void @__kmpc_taskloop(%ident_t* %{{.+}}, i32 [[TMP0]], i8* [[TMP65]], i32 1, i64* [[TMP84]], i64* [[TMP85]], i64 [[TMP89]], i32 0, i32 0, i64 0, i8* null)
// CHECK:    call void @__kmpc_end_taskgroup(%ident_t*

// CHECK:    ret i32

// CHECK: define internal void [[RED_INIT1]](i8*)
// CHECK: store float 0.000000e+00, float* %
// CHECK: ret void

// CHECK: define internal void [[RED_COMB1]](i8*, i8*)
// CHECK: fadd float %6, %7
// CHECK: store float %{{.+}}, float* %
// CHECK: ret void

// CHECK: define internal void [[RED_INIT2]](i8*)
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(
// CHECK: ret void

// CHECK: define internal void [[RED_FINI2]](i8*)
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: call void @
// CHECK: ret void

// CHECK: define internal void [[RED_COMB2]](i8*, i8*)
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: fadd float %
// CHECK: store float %{{.+}}, float* %
// CHECK: ret void

// CHECK: define internal void [[RED_INIT3]](i8*)
// CHECK: store float 0.000000e+00, float* %
// CHECK: ret void

// CHECK: define internal void [[RED_COMB3]](i8*, i8*)
// CHECK: fadd float %
// CHECK: store float %{{.+}}, float* %
// CHECK: ret void

// CHECK: define internal void [[RED_INIT4]](i8*)
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: store float 0.000000e+00, float* %
// CHECK: ret void

// CHECK: define internal void [[RED_COMB4]](i8*, i8*)
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: fadd float %
// CHECK: store float %{{.+}}, float* %
// CHECK: ret void

