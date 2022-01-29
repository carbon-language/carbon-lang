// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK1

// CK1: [[ST:%.+]] = type { i32, double* }
template <typename T>
struct ST {
  T a;
  double *b;
};

ST<int> gb;
double gc[100];

// CK1: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, i8* }
// CK1: [[KMP_TASK_T_WITH_PRIVATES:%.+]] = type { [[KMP_TASK_T:%[^,]+]], [[KMP_PRIVATES_T:%.+]] }
// CK1: [[KMP_TASK_T]] = type { i8*, i32 (i32, i8*)*, i32, %{{[^,]+}}, %{{[^,]+}} }
// CK1-32: [[KMP_PRIVATES_T]] = type { [1 x i64], [1 x i8*], [1 x i8*] }
// CK1-64: [[KMP_PRIVATES_T]] = type { [1 x i8*], [1 x i8*], [1 x i64] }

// CK1: [[SIZE00:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// CK1: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 2]

// CK1: [[SIZE02:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// CK1: [[MTYPE02:@.+]] = {{.+}}constant [1 x i64] zeroinitializer

// CK1: [[MTYPE03:@.+]] = {{.+}}constant [1 x i64] [i64 6]

// CK1: [[SIZE04:@.+]] = {{.+}}constant [2 x i64] [i64 sdiv exact (i64 sub (i64 ptrtoint (double** getelementptr (double*, double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1), i32 1) to i64), i64 ptrtoint (double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1) to i64)), i64 ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)), i64 24]
// CK1: [[MTYPE04:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976710672]

// CK1: [[MTYPE05:@.+]] = {{.+}}constant [1 x i64] [i64 1026]

// CK1: [[MTYPE06:@.+]] = {{.+}}constant [1 x i64] [i64 1030]

// CK1-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // Region 00
  // CK1-NOT: __tgt_target_data_begin
  // CK1-DAG: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{[^,]+}}, i32 %{{[^,]+}}, i8* [[TASK:%.+]])
  // CK1-DAG: [[TASK]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i[[sz:32|64]] {{36|64}}, i{{32|64}} 4, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES]]*)* [[OMP_TASK_ENTRY:@[^,]+]] to i32 (i32, i8*)*), i64 [[DEV:%.+]])
  // CK1-DAG: [[DEV]] = sext i32 [[DEV32:%.+]] to i64
  // CK1-DAG: [[TASK_WITH_PRIVATES:%.+]] = bitcast i8* [[TASK]] to [[KMP_TASK_T_WITH_PRIVATES]]*
  // CK1-DAG: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES]], [[KMP_TASK_T_WITH_PRIVATES]]* [[TASK_WITH_PRIVATES]], i32 0, i32 1
  // CK1-32-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 1
  // CK1-64-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 0
  // CK1-DAG: [[FPBPADDR:%.+]] = bitcast [1 x i8*]* [[FPBPGEP]] to i8*
  // CK1-DAG: [[BPADDR:%.+]] = bitcast i8** [[BPGEP:%.+]] to i8*
  // CK1-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPBPADDR]], i8* align {{4|8}} [[BPADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], i32 0, i32 0
  // CK1-DAG: [[BPGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
  // CK1-DAG: [[BPADDR:%.+]] = bitcast i8** [[BPGEP]] to [100 x double]**
  // CK1-DAG: store [100 x double]* [[GC:@[^,]+]], [100 x double]** [[BPADDR]], align
  // CK1-32-DAG: [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 2
  // CK1-64-DAG: [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 1
  // CK1-DAG: [[FPPADDR:%.+]] = bitcast [1 x i8*]* [[FPPGEP]] to i8*
  // CK1-DAG: [[PADDR:%.+]] = bitcast i8** [[PGEP:%.+]] to i8*
  // CK1-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPPADDR]], i8* align {{4|8}} [[PADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK1-DAG: [[PGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], i32 0, i32 0
  // CK1-DAG: [[PGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
  // CK1-DAG: [[PADDR:%.+]] = bitcast i8** [[PGEP]] to [100 x double]**
  // CK1-DAG: store [100 x double]* [[GC]], [100 x double]** [[PADDR]], align
  // CK1-32-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 0
  // CK1-64-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 2
  // CK1-DAG: [[FPSZADDR:%.+]] = bitcast [1 x i64]* [[FPSZGEP]] to i8*
  // CK1-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPSZADDR]], i8* align {{4|8}} bitcast ([1 x i64]* [[SIZE00]] to i8*), i[[sz]] {{4|8}}, i1 false)
  #pragma omp target exit data if(1+3-5) device(arg) map(from: gc) nowait
  {++arg;}

  // Region 01
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(release: la) if(1+3-4)
  {++arg;}

  // Region 02
  // CK1-NOT: __tgt_target_data_begin
  // CK1: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CK1: [[IFTHEN]]
  // CK1-DAG: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 4, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK1-DAG: store i32* [[VAL0:%[^,]+]], i32** [[CBP0]]
  // CK1-DAG: store i32* [[VAL0]], i32** [[CP0]]
  // CK1: br label %[[IFEND:[^,]+]]

  // CK1: [[IFELSE]]
  // CK1: br label %[[IFEND]]
  // CK1: [[IFEND]]
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(release: arg) if(arg) device(4)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 03
  // CK1-NOT: __tgt_target_data_begin
  // CK1-DAG: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK1-DAG: store float* [[VAL0:%[^,]+]], float** [[CBP0]]
  // CK1-DAG: store float* [[VAL0]], float** [[CP0]]
  // CK1-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(always, from: lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 04
  // CK1-NOT: __tgt_target_data_begin
  // CK1-DAG: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE04]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double***
  // CK1-DAG: store [[ST]]* @gb, [[ST]]** [[CBP0]]
  // CK1-DAG: store double** getelementptr inbounds ([[ST]], [[ST]]* @gb, i32 0, i32 1), double*** [[CP0]]


  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double***
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK1-DAG: store double** getelementptr inbounds ([[ST]], [[ST]]* @gb, i32 0, i32 1), double*** [[CBP1]]
  // CK1-DAG: store double* [[SEC1:%[^,]+]], double** [[CP1]]
  // CK1-DAG: [[SEC1]] = getelementptr inbounds {{.+}}double* [[SEC11:%[^,]+]], i{{.+}} 0
  // CK1-DAG: [[SEC11]] = load double*, double** getelementptr inbounds ([[ST]], [[ST]]* @gb, i32 0, i32 1),

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(release: gb.b[:3])
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 05
  // CK1-NOT: __tgt_target_data_begin
  // CK1-DAG: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK1-DAG: store float* [[VAL0:%[^,]+]], float** [[CBP0]]
  // CK1-DAG: store float* [[VAL0]], float** [[CP0]]
  // CK1-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(close, from: lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 06
  // CK1-NOT: __tgt_target_data_begin
  // CK1-DAG: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE06]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK1-DAG: store float* [[VAL0:%[^,]+]], float** [[CBP0]]
  // CK1-DAG: store float* [[VAL0]], float** [[CP0]]
  // CK1-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(always close, from: lb)
  {++arg;}
}

// CK1:     define internal {{.*}}i32 [[OMP_TASK_ENTRY]](i32 {{.*}}%{{[^,]+}}, [[KMP_TASK_T_WITH_PRIVATES]]* noalias %{{[^,]+}})
// CK1-DAG: call void @__tgt_target_data_end_nowait_mapper(%struct.ident_t* @{{.+}}, i64 %{{[^,]+}}, i32 1, i8** [[BP:%[^,]+]], i8** [[P:%[^,]+]], i64* [[SZ:%[^,]+]], i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[MTYPE00]], i32 0, i32 0), i8** null, i8** null)
// CK1-DAG: [[BP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK1-DAG: [[P]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK1-DAG: [[SZ]] = getelementptr inbounds [1 x i64], [1 x i64]* [[SZADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK1-DAG: [[BPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPBPADDR:%[^,]+]], align
// CK1-DAG: [[PADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPADDR:%[^,]+]], align
// CK1-DAG: [[SZADDR]] = load [1 x i64]*, [1 x i64]** [[FPSZADDR:%[^,]+]], align
// CK1-DAG: [[FN:%.+]] = bitcast void (i8*, ...)* {{%.*}} to void (i8*,
// CK1-DAG: call void [[FN]](i8* %{{[^,]+}}, [1 x i8*]** [[FPBPADDR]], [1 x i8*]** [[FPPADDR]], [1 x i64]** [[FPSZADDR]])
// CK1:     ret i32 0
// CK1:     }

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2: [[ST:%.+]] = type { i32, double* }
template <typename T>
struct ST {
  T a;
  double *b;

  T foo(T arg) {
    // Region 00
    #pragma omp target exit data map(always, release: b[1:3]) if(a>123) device(arg)
    {arg++;}
    return arg;
  }
};

// CK2: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976710676]

// CK2-LABEL: _Z3bari
int bar(int arg){
  ST<int> A;
  return A.foo(arg);
}

// Region 00
// CK2-NOT: __tgt_target_data_begin
// CK2: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK2: [[IFTHEN]]
// CK2-DAG: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 [[DEV:%[^,]+]], i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz:.+]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
// CK2-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK2-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
// CK2-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK2-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK2-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
// CK2-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double***
// CK2-DAG: store [[ST]]* [[VAR0:%[^,]+]], [[ST]]** [[CBP0]]
// CK2-DAG: store double** [[SEC0:%[^,]+]], double*** [[CP0]]
// CK2-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
// CK2-DAG: [[SEC0]] = getelementptr inbounds {{.*}}[[ST]]* [[VAR0]], i32 0, i32 1

// CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double***
// CK2-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
// CK2-DAG: store double** [[SEC0]], double*** [[CBP1]]
// CK2-DAG: store double* [[SEC1:%[^,]+]], double** [[CP1]]
// CK2-DAG: [[SEC1]] = getelementptr inbounds {{.*}}double* [[SEC11:%[^,]+]], i{{.+}} 1
// CK2-DAG: [[SEC11]] = load double*, double** [[SEC111:%[^,]+]],
// CK2-DAG: [[SEC111]] = getelementptr inbounds {{.*}}[[ST]]* [[VAR0]], i32 0, i32 1

// CK2: br label %[[IFEND:[^,]+]]

// CK2: [[IFELSE]]
// CK2: br label %[[IFEND]]
// CK2: [[IFEND]]
// CK2: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -verify -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
#ifdef CK3

// CK3-LABEL: no_target_devices
void no_target_devices(int arg) {
  // CK3-NOT: tgt_target_data_begin
  // CK3: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK3-NOT: tgt_target_data_end
  // CK3: ret
  #pragma omp target exit data map(from: arg) if(arg) device(4)
  {++arg;}
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-32

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK4

// CK4: [[STT:%.+]] = type { i32, double* }
template <typename T>
struct STT {
  T a;
  double *b;

  T foo(T arg) {
    // Region 00
    #pragma omp target exit data map(always close, release: b[1:3]) if(a>123) device(arg)
    {arg++;}
    return arg;
  }
};

// CK4: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976711700]

// CK4-LABEL: _Z3bari
int bar(int arg){
  STT<int> A;
  return A.foo(arg);
}

// Region 00
// CK4-NOT: __tgt_target_data_begin
// CK4: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK4: [[IFTHEN]]
// CK4-DAG: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 [[DEV:%[^,]+]], i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz:.+]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
// CK4-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK4-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
// CK4-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK4-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK4-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK4-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[STT]]**
// CK4-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double***
// CK4-DAG: store [[STT]]* [[VAR0:%[^,]+]], [[STT]]** [[CBP0]]
// CK4-DAG: store double** [[SEC0:%[^,]+]], double*** [[CP0]]
// CK4-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
// CK4-DAG: [[SEC0]] = getelementptr inbounds {{.*}}[[STT]]* [[VAR0]], i32 0, i32 1

// CK4-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK4-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK4-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double***
// CK4-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
// CK4-DAG: store double** [[SEC0]], double*** [[CBP1]]
// CK4-DAG: store double* [[SEC1:%[^,]+]], double** [[CP1]]
// CK4-DAG: [[SEC1]] = getelementptr inbounds {{.*}}double* [[SEC11:%[^,]+]], i{{.+}} 1
// CK4-DAG: [[SEC11]] = load double*, double** [[SEC111:%[^,]+]],
// CK4-DAG: [[SEC111]] = getelementptr inbounds {{.*}}[[STT]]* [[VAR0]], i32 0, i32 1

// CK4: br label %[[IFEND:[^,]+]]

// CK4: [[IFELSE]]
// CK4: br label %[[IFEND]]
// CK4: [[IFEND]]
// CK4: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
#endif
#endif
