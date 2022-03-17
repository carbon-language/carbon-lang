// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP45
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP45

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP50
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP50
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP50
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP50

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[KMP_TASK_T_WITH_PRIVATES:%.+]] = type { [[KMP_TASK_T:%.+]], [[KMP_PRIVATES_T:%.+]] }
// CHECK-DAG: [[KMP_TASK_T]] = type { i8*, i32 (i32, i8*)*, i32, {{%.+}}, {{%.+}} }
// CHECK-DAG: [[TT:%.+]] = type { i64, i8 }
// CHECK-DAG: [[S1:%.+]] = type { double }
// CHECK-DAG: [[S2:%.+]] = type { i32, i32, i32 }
// CHECK-DAG: [[ENTTY:%.+]] = type { i8*, i8*, i[[SZ:32|64]], i32, i32 }
// CHECK-DAG: [[ANON_T:%.+]] = type { i[[SZ]]*, i32, i32 }
// CHECK-32-DAG: [[KMP_PRIVATES_T]] = type { [2 x i64], i32*, i32, [2 x i8*], [2 x i8*] }
// CHECK-64-DAG: [[KMP_PRIVATES_T]] = type { i64*, [2 x i8*], [2 x i8*], [2 x i64], i32 }

// TCHECK: [[ENTTY:%.+]] = type { i8*, i8*, i{{32|64}}, i32, i32 }

// We have 9 target regions, but only 8 that actually will generate offloading
// code and have mapped arguments, and only 6 have all-constant map sizes.

// CHECK-DAG: [[SIZET:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 4]
// CHECK-DAG: [[MAPT:@.+]] = private unnamed_addr constant [2 x i64] [i64 544, i64 800]
// CHECK-DAG: [[SIZET2:@.+]] = private unnamed_addr constant [1 x i{{32|64}}] [i64 2]
// CHECK-DAG: [[MAPT2:@.+]] = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: [[SIZET3:@.+]] = private unnamed_addr constant [2 x i64] [i64 4, i64 2]
// CHECK-DAG: [[MAPT3:@.+]] = private unnamed_addr constant [2 x i64] [i64 800, i64 800]
// CHECK-DAG: [[MAPT4:@.+]] = private unnamed_addr constant [9 x i64] [i64 800, i64 547, i64 800, i64 547, i64 547, i64 800, i64 800, i64 547, i64 547]
// CHECK-DAG: [[SIZET5:@.+]] = private unnamed_addr constant [3 x i64] [i64 4, i64 2, i64 40]
// CHECK-DAG: [[MAPT5:@.+]] = private unnamed_addr constant [3 x i64] [i64 800, i64 800, i64 547]
// CHECK-DAG: [[SIZET6:@.+]] = private unnamed_addr constant [4 x i64] [i64 4, i64 2, i64 1, i64 40]
// CHECK-DAG: [[MAPT6:@.+]] = private unnamed_addr constant [4 x i64] [i64 800, i64 800, i64 800, i64 547]
// CHECK-DAG: [[MAPT7:@.+]] = private unnamed_addr constant [5 x i64] [i64 547, i64 800, i64 800, i64 800, i64 547]
// CHECK-DAG: [[SIZET9:@.+]] = private unnamed_addr constant [1 x i64] [i64 12]
// CHECK-DAG: [[MAPT10:@.+]] = private unnamed_addr constant [1 x i64] [i64 35]
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0

// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK-NOT: @{{.+}} = weak constant [[ENTTY]]

// Check target registration is registered as a Ctor.
// CHECK: appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* @.omp_offloading.requires_reg, i8* null }]


template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
};

int global;
extern int global;

// CHECK: define {{.*}}[[FOO:@.+]](
int foo(int n) {
  // CHECK: [[OFFLOADBPTR:%.+]] = alloca [2 x i8*], align
  // CHECK: [[OFFLOADPTR:%.+]] = alloca [2 x i8*], align
  // CHECK: [[OFFLOADMAPPER:%.+]] = alloca [2 x i8*], align
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;
  static long *plocal;

  // CHECK:       [[ADD:%.+]] = add nsw i32
  // CHECK:       store i32 [[ADD]], i32* [[DEVICE_CAP:%.+]],
  // CHECK:       [[DEV:%.+]] = load i32, i32* [[DEVICE_CAP]],
  // CHECK:       [[DEVICE:%.+]] = sext i32 [[DEV]] to i64
  // CHECK:       [[RET:%.+]] = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 [[DEVICE]], i8* @{{[^,]+}}, i32 0, i8** null, i8** null, i64* null, i64* null, i8** null, i8** null)
  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT0:@.+]]()
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target device(global + a)
  {
  }

  // CHECK: [[BPRGEP:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OFFLOADBPTR]], i32 0, i32 0
  // CHECK: [[PRGEP:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OFFLOADPTR]], i32 0, i32 0
  // CHECK: [[BPRGEP:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OFFLOADBPTR]], i32 0, i32 0
  // CHECK: [[PRGEP:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OFFLOADPTR]], i32 0, i32 0
  // CHECK: [[DEVICE:%.+]] = sext i32 {{%.+}} to i64
  // CHECK-32: [[TASK:%.+]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* {{.+}}, i32 %0, i32 1, i32 60, i32 12, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES]]*)* [[OMP_TASK_ENTRY:@.+]] to i32 (i32, i8*)*), i64 [[DEVICE]])
  // CHECK-64: [[TASK:%.+]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* {{.+}}, i32 %0, i32 1, i64 104, i64 16, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES]]*)* [[OMP_TASK_ENTRY:@.+]] to i32 (i32, i8*)*), i64 [[DEVICE]])
  // CHECK: [[TASK_WITH_PRIVATES:%.+]] = bitcast i8* [[TASK]] to [[KMP_TASK_T_WITH_PRIVATES]]*
  // CHECK: [[TASK_WITH_PRIVATES_GEP:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES]], [[KMP_TASK_T_WITH_PRIVATES]]* [[TASK_WITH_PRIVATES]], i32 0, i32 1
  // CHECK-32: [[SIZEGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 0
  // CHECK-32: [[SIZEADDR:%.+]] = bitcast [2 x i64]* [[SIZEGEP]] to i8*
  // CHECK-32: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 [[SIZEADDR]], i8* align 4 bitcast ([2 x i64]* [[SIZET]] to i8*), i32 16, i1 false)
  // CHECK-32: [[FPBPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 3
  // CHECK-32: [[FPBPRCAST:%.+]] = bitcast [2 x i8*]* [[FPBPRGEP]] to i8*
  // CHECK-32: [[BPRCAST:%.+]] = bitcast i8** [[BPRGEP]] to i8*
  // CHECK-32: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 [[FPBPRCAST]], i8* align 4 [[BPRCAST]], i32 8, i1 false)
  // CHECK-32: [[FPPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 4
  // CHECK-32: [[FPPRCAST:%.+]] = bitcast [2 x i8*]* [[FPPRGEP]] to i8*
  // CHECK-32: [[PRCAST:%.+]] = bitcast i8** [[PRGEP]] to i8*
  // CHECK-32: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 [[FPPRCAST]], i8* align 4 [[PRCAST]], i32 8, i1 false)
  // CHECK-64: [[FPBPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 1
  // CHECK-64: [[FPBPRCAST:%.+]] = bitcast [2 x i8*]* [[FPBPRGEP]] to i8*
  // CHECK-64: [[BPR_CAST:%.+]] = bitcast i8** [[BPRGEP]] to i8*
  // CHECK-64: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[FPBPRCAST]], i8* align 8 [[BPR_CAST]], i64 16, i1 false)
  // CHECK-64: [[FPPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 2
  // CHECK-64: [[FPPRCAST:%.+]] = bitcast [2 x i8*]* [[FPPRGEP]] to i8*
  // CHECK-64: [[PR_CAST:%.+]] = bitcast i8** [[PRGEP]] to i8*
  // CHECK-64: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[FPPRCAST]], i8* align 8 [[PR_CAST]], i64 16, i1 false)
  // CHECK-64: [[SIZEGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 3
  // CHECK-64: [[SIZE_CAST:%.+]] = bitcast [2 x i64]* [[SIZEGEP]] to i8*
  // CHECK-64: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[SIZE_CAST]], i8* align 8 bitcast ([2 x i64]* [[SIZET]] to i8*), i64 16, i1 false)
  // CHECK: call i32 @__kmpc_omp_task([[IDENT_T]]* {{.+}}, i32 {{.+}}, i8* [[TASK]])
  #pragma omp target device(global + a) nowait
  {
    static int local1;
    *plocal = global;
    local1 = global;
  }

  // CHECK:       call void [[HVT1:@.+]](i[[SZ]] {{[^,]+}})
  #pragma omp target if(0) firstprivate(global)
  {
    global += 1;
  }

  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 1, i8** [[BP:%[^,]+]], i8** [[P:%[^,]+]], i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[SIZET2]], i32 0, i32 0), i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[MAPT2]], i32 0, i32 0), i8** null, i8** null)
  // CHECK-DAG:   [[BP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[P]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPR]], i32 0, i32 [[IDX0:[0-9]+]]
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PR]], i32 0, i32 [[IDX0]]
  // CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[BP0:%[^,]+]], i[[SZ]]* [[CBPADDR0]]
  // CHECK-DAG:   store i[[SZ]] [[P0:%[^,]+]], i[[SZ]]* [[CPADDR0]]

  // CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT2:@.+]](i[[SZ]] {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target if(1)
  {
    aa += 1;
  }

  // CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 10
  // CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CHECK:       [[IFTHEN]]
  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 2, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[SIZET3]], i32 0, i32 0), i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[MAPT3]], i32 0, i32 0), i8** null, i8** null)
  // CHECK-DAG:   [[BPR]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[PR]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P:%[^,]+]], i32 0, i32 0

  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP]], i32 0, i32 0
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P]], i32 0, i32 0
  // CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[BP0:%[^,]+]], i[[SZ]]* [[CBPADDR0]]
  // CHECK-DAG:   store i[[SZ]] [[P0:%[^,]+]], i[[SZ]]* [[CPADDR0]]

  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP]], i32 0, i32 1
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P]], i32 0, i32 1
  // CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[BP1:%[^,]+]], i[[SZ]]* [[CBPADDR1]]
  // CHECK-DAG:   store i[[SZ]] [[P1:%[^,]+]], i[[SZ]]* [[CPADDR1]]
  // CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT3:@.+]]({{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  // CHECK-NEXT:  br label %[[IFEND:.+]]
  // CHECK:       [[IFELSE]]
  // CHECK:       call void [[HVT3]]({{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[IFEND]]

  // CHECK:       [[IFEND]]
  #pragma omp target if(n>10)
  {
    a += 1;
    aa += 1;
  }

  // We capture 3 VLA sizes in this target region
  // CHECK-64:       [[A_VAL:%.+]] = load i32, i32* %{{.+}},
  // CHECK-64:       [[A_ADDR:%.+]] = bitcast i[[SZ]]* [[A_CADDR:%.+]] to i32*
  // CHECK-64:       store i32 [[A_VAL]], i32* [[A_ADDR]],
  // CHECK-64:       [[A_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[A_CADDR]],

  // CHECK-32:       [[A_VAL:%.+]] = load i32, i32* %{{.+}},
  // CHECK-32:       store i32 [[A_VAL]], i32* [[A_CADDR:%.+]],
  // CHECK-32:       [[A_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[A_CADDR]],

  // CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 20
  // CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[IFELSE:[^,]+]]
  // CHECK:       [[TRY]]
  // CHECK-64:    [[BNSIZE:%.+]] = mul nuw i64 [[VLA0:%.+]], 4
  // CHECK-32:    [[BNSZSIZE:%.+]] = mul nuw i32 [[VLA0:%.+]], 4
  // CHECK-32:    [[BNSIZE:%.+]] = sext i32 [[BNSZSIZE]] to i64
  // CHECK:       [[CNELEMSIZE2:%.+]] = mul nuw i[[SZ]] 5, [[VLA1:%.+]]
  // CHECK-64:    [[CNSIZE:%.+]] = mul nuw i64 [[CNELEMSIZE2]], 8
  // CHECK-32:    [[CNSZSIZE:%.+]] = mul nuw i32 [[CNELEMSIZE2]], 8
  // CHECK-32:    [[CNSIZE:%.+]] = sext i32 [[CNSZSIZE]] to i64

  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 9, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* [[SR:%[^,]+]], i64* getelementptr inbounds ([9 x i64], [9 x i64]* [[MAPT4]], i32 0, i32 0), i8** null, i8** null)
  // CHECK-DAG:   [[BPR]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[PR]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[SR]] = getelementptr inbounds [9 x i64], [9 x i64]* [[S:%[^,]+]], i32 0, i32 0

  // CHECK-DAG:   [[SADDR0:%.+]] = getelementptr inbounds [9 x i64], [9 x i64]* [[S]], i32 0, i32 [[IDX0:0]]
  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX0]]
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX0]]
  // CHECK-DAG:   [[SADDR1:%.+]] = getelementptr inbounds [9 x i64], [9 x i64]* [[S]], i32 0, i32 [[IDX1:1]]
  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX1]]
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX1]]
  // CHECK-DAG:   [[SADDR2:%.+]] = getelementptr inbounds [9 x i64], [9 x i64]* [[S]], i32 0, i32 [[IDX2:2]]
  // CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX2]]
  // CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX2]]
  // CHECK-DAG:   [[SADDR3:%.+]] = getelementptr inbounds [9 x i64], [9 x i64]* [[S]], i32 0, i32 [[IDX3:3]]
  // CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX3]]
  // CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX3]]
  // CHECK-DAG:   [[SADDR4:%.+]] = getelementptr inbounds [9 x i64], [9 x i64]* [[S]], i32 0, i32 [[IDX4:4]]
  // CHECK-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX4]]
  // CHECK-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX4]]
  // CHECK-DAG:   [[SADDR5:%.+]] = getelementptr inbounds [9 x i64], [9 x i64]* [[S]], i32 0, i32 [[IDX5:5]]
  // CHECK-DAG:   [[BPADDR5:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX5]]
  // CHECK-DAG:   [[PADDR5:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX5]]
  // CHECK-DAG:   [[SADDR6:%.+]] = getelementptr inbounds [9 x i64], [9 x i64]* [[S]], i32 0, i32 [[IDX6:6]]
  // CHECK-DAG:   [[BPADDR6:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX6]]
  // CHECK-DAG:   [[PADDR6:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX6]]
  // CHECK-DAG:   [[SADDR7:%.+]] = getelementptr inbounds [9 x i64], [9 x i64]* [[S]], i32 0, i32 [[IDX7:7]]
  // CHECK-DAG:   [[BPADDR7:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX7]]
  // CHECK-DAG:   [[PADDR7:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX7]]
  // CHECK-DAG:   [[SADDR8:%.+]] = getelementptr inbounds [9 x i64], [9 x i64]* [[S]], i32 0, i32 [[IDX8:8]]
  // CHECK-DAG:   [[BPADDR8:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX8]]
  // CHECK-DAG:   [[PADDR8:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX8]]

  // The names below are not necessarily consistent with the names used for the
  // addresses above as some are repeated.
  // CHECK-DAG:   [[CBPADDR2:%.+]] = bitcast i8** [[BPADDR2]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR2:%.+]] = bitcast i8** [[PADDR2]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CBPADDR2]]
  // CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CPADDR2]]
  // CHECK-DAG:   store i64 {{4|8}}, i64* [[SADDR2]]

  // CHECK-DAG:   [[CBPADDR6:%.+]] = bitcast i8** [[BPADDR6]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR6:%.+]] = bitcast i8** [[PADDR6]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[VLA1]], i[[SZ]]* [[CBPADDR6]]
  // CHECK-DAG:   store i[[SZ]] [[VLA1]], i[[SZ]]* [[CPADDR6]]
  // CHECK-DAG:   store i64 {{4|8}}, i64* [[SADDR6]]

  // CHECK-DAG:   [[CBPADDR5:%.+]] = bitcast i8** [[BPADDR5]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR5:%.+]] = bitcast i8** [[PADDR5]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] 5, i[[SZ]]* [[CBPADDR5]]
  // CHECK-DAG:   store i[[SZ]] 5, i[[SZ]]* [[CPADDR5]]
  // CHECK-DAG:   store i64 {{4|8}}, i64* [[SADDR5]]

  // CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[A_CVAL]], i[[SZ]]* [[CBPADDR0]]
  // CHECK-DAG:   store i[[SZ]] [[A_CVAL]], i[[SZ]]* [[CPADDR0]]
  // CHECK-DAG:   store i64 4, i64* [[SADDR0]]

  // CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to [10 x float]**
  // CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to [10 x float]**
  // CHECK-DAG:   store [10 x float]* %{{.+}}, [10 x float]** [[CBPADDR1]]
  // CHECK-DAG:   store [10 x float]* %{{.+}}, [10 x float]** [[CPADDR1]]
  // CHECK-DAG:   store i64 40, i64* [[SADDR1]]

  // CHECK-DAG:   [[CBPADDR3:%.+]] = bitcast i8** [[BPADDR3]] to float**
  // CHECK-DAG:   [[CPADDR3:%.+]] = bitcast i8** [[PADDR3]] to float**
  // CHECK-DAG:   store float* %{{.+}}, float** [[CBPADDR3]]
  // CHECK-DAG:   store float* %{{.+}}, float** [[CPADDR3]]
  // CHECK-DAG:   store i64 [[BNSIZE]], i64* [[SADDR3]]

  // CHECK-DAG:   [[CBPADDR4:%.+]] = bitcast i8** [[BPADDR4]] to [5 x [10 x double]]**
  // CHECK-DAG:   [[CPADDR4:%.+]] = bitcast i8** [[PADDR4]] to [5 x [10 x double]]**
  // CHECK-DAG:   store [5 x [10 x double]]* %{{.+}}, [5 x [10 x double]]** [[CBPADDR4]]
  // CHECK-DAG:   store [5 x [10 x double]]* %{{.+}}, [5 x [10 x double]]** [[CPADDR4]]
  // CHECK-DAG:   store i64 400, i64* [[SADDR4]]

  // CHECK-DAG:   [[CBPADDR7:%.+]] = bitcast i8** [[BPADDR7]] to double**
  // CHECK-DAG:   [[CPADDR7:%.+]] = bitcast i8** [[PADDR7]] to double**
  // CHECK-DAG:   store double* %{{.+}}, double** [[CBPADDR7]]
  // CHECK-DAG:   store double* %{{.+}}, double** [[CPADDR7]]
  // CHECK-DAG:   store i64 [[CNSIZE]], i64* [[SADDR7]]

  // CHECK-DAG:   [[CBPADDR8:%.+]] = bitcast i8** [[BPADDR8]] to [[TT]]**
  // CHECK-DAG:   [[CPADDR8:%.+]] = bitcast i8** [[PADDR8]] to [[TT]]**
  // CHECK-DAG:   store [[TT]]* %{{.+}}, [[TT]]** [[CBPADDR8]]
  // CHECK-DAG:   store [[TT]]* %{{.+}}, [[TT]]** [[CPADDR8]]
  // CHECK-DAG:   store i64 {{12|16}}, i64* [[SADDR8]]

  // CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT4:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  // CHECK-NEXT:  br label %[[IFEND:.+]]
  // CHECK:       [[IFELSE]]
  // CHECK:       call void [[HVT4]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[IFEND]]

  // CHECK:       [[IFEND]]
  #pragma omp target if(n>20)
  {
    a += 1;
    b[2] += 1.0;
    bn[3] += 1.0;
    c[1][2] += 1.0;
    cn[1][3] += 1.0;
    d.X += 1;
    d.Y += 1;
  }

  return a;
}

// Check that the offloading functions are emitted and that the arguments are
// correct and loaded correctly for the target regions in foo().

// CHECK:       define internal void [[HVT0]]()

// CHECK: define internal void [[HVT0_:@.+]](i[[SZ]]* noundef {{%[^,]+}}, i[[SZ]] noundef {{%[^,]+}})
// CHECK: define internal {{.*}}i32 [[OMP_TASK_ENTRY]](i32 {{.*}}%0, [[KMP_TASK_T_WITH_PRIVATES]]* noalias noundef %1)
// CHECK-DAG: [[RET:%.+]] = call i32 @__tgt_target_nowait_mapper(%struct.ident_t* @{{.+}}, i64 [[DEVICE:%.+]], i8* @{{[^,]+}}, i32 2, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* [[SIZE:%.+]], i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[MAPT]], i32 0, i32 0), i8** null, i8** null, i32 0, i8* null, i32 0, i8* null)
// CHECK-DAG: [[DEVICE]] = sext i32 [[DEV:%.+]] to i64
// CHECK-DAG: [[DEV]] = load i32, i32* [[DEVADDR:%.+]], align
// CHECK-DAG: [[DEVADDR]] = getelementptr inbounds [[ANON_T]], [[ANON_T]]* {{%.+}}, i32 0, i32 2
// CHECK-DAG: [[BPR]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BPRADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CHECK-DAG: [[PR]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PRADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CHECK-DAG: [[SIZE]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZEADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CHECK-DAG: [[BPRADDR]] = load [2 x i8*]*, [2 x i8*]** [[FPPTR_BPR:%.+]], align
// CHECK-DAG: [[PRADDR]] = load [2 x i8*]*, [2 x i8*]** [[FPPTR_PR:%.+]], align
// CHECK-DAG: [[SIZEADDR]] = load [2 x i64]*, [2 x i64]** [[FPPTR_SIZE:%.+]], align
// CHECK-DAG: [[FN:%.+]] = bitcast void (i8*, ...)* {{%[0-9]+}} to void (i8*,
// CHECK-DAG: call void [[FN]](i8* {{%[^,]+}}, i[[SZ]]*** [[FPPTR_PLOCAL:%.+]], i32** [[FPPTR_GLOBAL:%.+]], [2 x i8*]** [[FPPTR_BPR]], [2 x i8*]** [[FPPTR_PR]], [2 x i64]** [[FPPTR_SIZE]])
// CHECK-DAG: [[PLOCALADDR:%.+]] = load i[[SZ]]**, i[[SZ]]*** [[FPPTR_PLOCAL]], align
// CHECK-DAG: {{%.+}} = load i32*, i32** [[FPPTR_GLOBAL:%.+]], align
// CHECK: [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT: br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// CHECK: [[FAIL]]
// CHECK: [[PLOCAL:%.+]] = load i[[SZ]]*, i[[SZ]]** [[PLOCALADDR]], align
// CHECK: [[GLOBAL:%.+]] = load i32, i32* {{@.+}}, align
// CHECK-64: [[CONVI:%.+]] = bitcast i64* [[GLOBALCAST:%.+]] to i32*
// CHECK-32: store i32 [[GLOBAL]], i32* [[GLOBALCAST:%.+]], align
// CHECK-64: store i32 [[GLOBAL]], i32* [[CONVI]], align
// CHECK: [[GLOBAL:%.+]] = load i[[SZ]], i[[SZ]]* [[GLOBALCAST]], align
// CHECK: call void [[HVT0_]](i[[SZ]]* [[PLOCAL]], i[[SZ]] [[GLOBAL]])
// CHECK-NEXT: br label %[[END]]
// CHECK: [[END]]

// CHECK:       define internal void [[HVT1]](i[[SZ]] noundef %{{.+}})
// Create stack storage and store argument in there.
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK-64:    [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i32*
// CHECK-64:    load i32, i32* [[AA_CADDR]], align
// CHECK-32:    load i32, i32* [[AA_ADDR]], align

// CHECK:       define internal void [[HVT2]](i[[SZ]] noundef %{{.+}})
// Create stack storage and store argument in there.
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK:       [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i16*
// CHECK:       load i16, i16* [[AA_CADDR]], align

// CHECK:       define internal void [[HVT3]]
// Create stack storage and store argument in there.
// CHECK:       [[A_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[A_ADDR]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK-64-DAG:[[A_CADDR:%.+]] = bitcast i[[SZ]]* [[A_ADDR]] to i32*
// CHECK-DAG:   [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i16*
// CHECK-64-DAG:load i32, i32* [[A_CADDR]], align
// CHECK-32-DAG:load i32, i32* [[A_ADDR]], align
// CHECK-DAG:   load i16, i16* [[AA_CADDR]], align

// CHECK:       define internal void [[HVT4]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca [10 x float]*
// CHECK:       [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_BN:%.+]] = alloca float*
// CHECK:       [[LOCAL_C:%.+]] = alloca [5 x [10 x double]]*
// CHECK:       [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA3:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_CN:%.+]] = alloca double*
// CHECK:       [[LOCAL_D:%.+]] = alloca [[TT]]*
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// CHECK-DAG:   store [10 x float]* [[ARG_B:%.+]], [10 x float]** [[LOCAL_B]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA1:%.+]], i[[SZ]]* [[LOCAL_VLA1]]
// CHECK-DAG:   store float* [[ARG_BN:%.+]], float** [[LOCAL_BN]]
// CHECK-DAG:   store [5 x [10 x double]]* [[ARG_C:%.+]], [5 x [10 x double]]** [[LOCAL_C]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA2:%.+]], i[[SZ]]* [[LOCAL_VLA2]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA3:%.+]], i[[SZ]]* [[LOCAL_VLA3]]
// CHECK-DAG:   store double* [[ARG_CN:%.+]], double** [[LOCAL_CN]]
// CHECK-DAG:   store [[TT]]* [[ARG_D:%.+]], [[TT]]** [[LOCAL_D]]

// CHECK-64-DAG:[[REF_A:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x float]*, [10 x float]** [[LOCAL_B]],
// CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA1]],
// CHECK-DAG:   [[REF_BN:%.+]] = load float*, float** [[LOCAL_BN]],
// CHECK-DAG:   [[REF_C:%.+]] = load [5 x [10 x double]]*, [5 x [10 x double]]** [[LOCAL_C]],
// CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA2]],
// CHECK-DAG:   [[VAL_VLA3:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA3]],
// CHECK-DAG:   [[REF_CN:%.+]] = load double*, double** [[LOCAL_CN]],
// CHECK-DAG:   [[REF_D:%.+]] = load [[TT]]*, [[TT]]** [[LOCAL_D]],

// Use captures.
// CHECK-64-DAG:   load i32, i32* [[REF_A]]
// CHECK-32-DAG:   load i32, i32* [[LOCAL_A]]
// CHECK-DAG:   getelementptr inbounds [10 x float], [10 x float]* [[REF_B]], i[[SZ]] 0, i[[SZ]] 2
// CHECK-DAG:   getelementptr inbounds float, float* [[REF_BN]], i[[SZ]] 3
// CHECK-DAG:   getelementptr inbounds [5 x [10 x double]], [5 x [10 x double]]* [[REF_C]], i[[SZ]] 0, i[[SZ]] 1
// CHECK-DAG:   getelementptr inbounds double, double* [[REF_CN]], i[[SZ]] %{{.+}}
// CHECK-DAG:   getelementptr inbounds [[TT]], [[TT]]* [[REF_D]], i32 0, i32 0

template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target if(n>40)
  {
    a += 1;
    aa += 1;
    b[2] += 1;
  }

  return a;
}

static
int fstatic(int n) {
  int a = 0;
  short aa = 0;
  char aaa = 0;
  int b[10];

  #pragma omp target if(n>50)
  {
    a += 1;
    aa += 1;
    aaa += 1;
    b[2] += 1;
  }

  return a;
}

struct S1 {
  double a;

  int r1(int n){
    int b = n+1;
    short int c[2][n];

    #pragma omp target if(n>60)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

    return c[1][1] + (int)b;
  }
};

// CHECK: define {{.*}}@{{.*}}bar{{.*}}
int bar(int n){
  int a = 0;

  // CHECK: call {{.*}}i32 [[FOO]](i32 {{.*}})
  a += foo(n);

  S1 S;
  // CHECK: call {{.*}}i32 [[FS1:@.+]]([[S1]]* {{.*}}, i32 {{.*}})
  a += S.r1(n);

  // CHECK: call {{.*}}i32 [[FSTATIC:@.+]](i32 {{.*}})
  a += fstatic(n);

  // CHECK: call {{.*}}i32 [[FTEMPLATE:@.+]](i32 {{.*}})
  a += ftemplate<int>(n);

  return a;
}

//
// CHECK: define {{.*}}[[FS1]]
//
// CHECK:          i8* @llvm.stacksave()
// CHECK-64:       [[B_ADDR:%.+]] = bitcast i[[SZ]]* [[B_CADDR:%.+]] to i32*
// CHECK-64:       store i32 %{{.+}}, i32* [[B_ADDR]],
// CHECK-64:       [[B_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[B_CADDR]],

// CHECK-32:       store i32 %{{.+}}, i32* %__vla_expr
// CHECK-32:       store i32 %{{.+}}, i32* [[B_ADDR:%.+]],
// CHECK-32:       [[B_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[B_ADDR]],

// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 60
// CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[TRY]]
// We capture 2 VLA sizes in this target region
// CHECK:       [[CELEMSIZE2:%.+]] = mul nuw i[[SZ]] 2, [[VLA0:%.+]]
// CHECK-64:    [[CSIZE:%.+]] = mul nuw i64 [[CELEMSIZE2]], 2
// CHECK-32:    [[CSZSIZE:%.+]] = mul nuw i32 [[CELEMSIZE2]], 2
// CHECK-32:    [[CSIZE:%.+]] = sext i32 [[CSZSIZE]] to i64

// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 5, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* [[SR:%[^,]+]], i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[MAPT7]], i32 0, i32 0), i8** null, i8** null)
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P:%.+]], i32 0, i32 0
// CHECK-DAG:   [[SR]] = getelementptr inbounds [5 x i64], [5 x i64]* [[S:%.+]], i32 0, i32 0
// CHECK-DAG:   [[SADDR0:%.+]] = getelementptr inbounds [5 x i64], [5 x i64]* [[S]], i32 0, i32 [[IDX0:0]]
// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 0, i32 [[IDX0]]
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 0, i32 [[IDX0]]
// CHECK-DAG:   [[SADDR1:%.+]] = getelementptr inbounds [5 x i64], [5 x i64]* [[S]], i32 0, i32 [[IDX1:1]]
// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 0, i32 [[IDX1]]
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 0, i32 [[IDX1]]
// CHECK-DAG:   [[SADDR2:%.+]] = getelementptr inbounds [5 x i64], [5 x i64]* [[S]], i32 0, i32 [[IDX2:2]]
// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 0, i32 [[IDX2]]
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 0, i32 [[IDX2]]
// CHECK-DAG:   [[SADDR3:%.+]] = getelementptr inbounds [5 x i64], [5 x i64]* [[S]], i32 0, i32 [[IDX3:3]]
// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 0, i32 [[IDX3]]
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 0, i32 [[IDX3]]
// CHECK-DAG:   [[SADDR4:%.+]] = getelementptr inbounds [5 x i64], [5 x i64]* [[S]], i32 0, i32 [[IDX4:4]]
// CHECK-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 0, i32 [[IDX4]]
// CHECK-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 0, i32 [[IDX4]]

// The names below are not necessarily consistent with the names used for the
// addresses above as some are repeated.
// CHECK-DAG:   [[CBPADDR4:%.+]] = bitcast i8** [[BPADDR4]] to i16**
// CHECK-DAG:   [[CPADDR4:%.+]] = bitcast i8** [[PADDR4]] to i16**
// CHECK-DAG:   store i16* %{{.+}}, i16** [[CBPADDR4]]
// CHECK-DAG:   store i16* %{{.+}}, i16** [[CPADDR4]]
// CHECK-DAG:   store i64 [[CSIZE]], i64* [[SADDR4]]

// CHECK-DAG:   [[CBPADDR3:%.+]] = bitcast i8** [[BPADDR3]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR3:%.+]] = bitcast i8** [[PADDR3]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CBPADDR3]]
// CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CPADDR3]]
// CHECK-DAG:   store i64 {{4|8}}, i64* [[SADDR3]]

// CHECK-DAG:   [[CBPADDR2:%.+]] = bitcast i8** [[BPADDR2]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR2:%.+]] = bitcast i8** [[PADDR2]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] 2, i[[SZ]]* [[CBPADDR2]]
// CHECK-DAG:   store i[[SZ]] 2, i[[SZ]]* [[CPADDR2]]
// CHECK-DAG:   store i64 {{4|8}}, i64* [[SADDR2]]

// CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[B_CVAL]], i[[SZ]]* [[CBPADDR1]]
// CHECK-DAG:   store i[[SZ]] [[B_CVAL]], i[[SZ]]* [[CPADDR1]]
// CHECK-DAG:   store i64 4, i64* [[SADDR1]]

// CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to [[S1]]**
// CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to double**
// CHECK-DAG:   store [[S1]]* [[THIS:%.+]], [[S1]]** [[CBPADDR0]]
// CHECK-DAG:   store double* [[A:%.+]], double** [[CPADDR0]]
// CHECK-DAG:   store i64 8, i64* [[SADDR0]]

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT7:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT7]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]

// CHECK:       [[IFEND]]

//
// CHECK: define {{.*}}[[FSTATIC]]
//
// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 50
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 4, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* getelementptr inbounds ([4 x i64], [4 x i64]* [[SIZET6]], i32 0, i32 0), i64* getelementptr inbounds ([4 x i64], [4 x i64]* [[MAPT6]], i32 0, i32 0), i8** null, i8** null)
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 0
// CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VAL0:%[^,]+]], i[[SZ]]* [[CBPADDR0]]
// CHECK-DAG:   store i[[SZ]] [[VAL0]], i[[SZ]]* [[CPADDR0]]

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 1
// CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VAL1:%[^,]+]], i[[SZ]]* [[CBPADDR1]]
// CHECK-DAG:   store i[[SZ]] [[VAL1]], i[[SZ]]* [[CPADDR1]]

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 2
// CHECK-DAG:   [[CBPADDR2:%.+]] = bitcast i8** [[BPADDR2]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR2:%.+]] = bitcast i8** [[PADDR2]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VAL2:%[^,]+]], i[[SZ]]* [[CBPADDR2]]
// CHECK-DAG:   store i[[SZ]] [[VAL2]], i[[SZ]]* [[CPADDR2]]

// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 3
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 3
// CHECK-DAG:   [[CBPADDR3:%.+]] = bitcast i8** [[BPADDR3]] to [10 x i32]**
// CHECK-DAG:   [[CPADDR3:%.+]] = bitcast i8** [[PADDR3]] to [10 x i32]**
// CHECK-DAG:   store [10 x i32]* [[VAL3:%[^,]+]], [10 x i32]** [[CBPADDR3]]
// CHECK-DAG:   store [10 x i32]* [[VAL3]], [10 x i32]** [[CPADDR3]]

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT6:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT6]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]

// CHECK:       [[IFEND]]

//
// CHECK: define {{.*}}[[FTEMPLATE]]
//
// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 40
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 3, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[SIZET5]], i32 0, i32 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[MAPT5]], i32 0, i32 0), i8** null, i8** null)
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 0
// CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VAL0:%[^,]+]], i[[SZ]]* [[CBPADDR0]]
// CHECK-DAG:   store i[[SZ]] [[VAL0]], i[[SZ]]* [[CPADDR0]]

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 1
// CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VAL1:%[^,]+]], i[[SZ]]* [[CBPADDR1]]
// CHECK-DAG:   store i[[SZ]] [[VAL1]], i[[SZ]]* [[CPADDR1]]

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 2
// CHECK-DAG:   [[CBPADDR2:%.+]] = bitcast i8** [[BPADDR2]] to [10 x i32]**
// CHECK-DAG:   [[CPADDR2:%.+]] = bitcast i8** [[PADDR2]] to [10 x i32]**
// CHECK-DAG:   store [10 x i32]* [[VAL2:%[^,]+]], [10 x i32]** [[CBPADDR2]]
// CHECK-DAG:   store [10 x i32]* [[VAL2]], [10 x i32]** [[CPADDR2]]

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT5:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT5]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]

// CHECK:       [[IFEND]]

// OMP45: define internal void @__omp_offloading_{{.+}}_{{.+}}bar{{.+}}_l{{[0-9]+}}(i[[SZ]] noundef %{{.+}})

// OMP45: define {{.*}}@{{.*}}zee{{.*}}

// OMP45:       [[LOCAL_THIS:%.+]] = alloca [[S2]]*
// OMP45:       [[BP:%.+]] = alloca [1 x i8*]
// OMP45:       [[P:%.+]] = alloca [1 x i8*]
// OMP45:       [[LOCAL_THIS1:%.+]] = load [[S2]]*, [[S2]]** [[LOCAL_THIS]]

// OMP45:       call void @__kmpc_critical(
// OMP45:       [[ARR_IDX:%.+]] = getelementptr inbounds [[S2]], [[S2]]* [[LOCAL_THIS1]], i[[SZ]] 0
// OMP45:       [[ARR_IDX2:%.+]] = getelementptr inbounds [[S2]], [[S2]]* [[LOCAL_THIS1]], i[[SZ]] 0

// OMP45-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
// OMP45-DAG:   [[PADDR0:%.+]] =  getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
// OMP45-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to [[S2]]**
// OMP45-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to [[S2]]**
// OMP45-DAG:   store [[S2]]* [[ARR_IDX]], [[S2]]** [[CBPADDR0]]
// OMP45-DAG:   store [[S2]]* [[ARR_IDX2]], [[S2]]** [[CPADDR0]]

// OMP45:       [[BPR:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
// OMP45:       [[PR:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
// OMP45:       [[RET:%.+]] = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 1, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[SIZET9]], i32 0, i32 0), i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[MAPT10]], i32 0, i32 0), i8** null, i8** null)
// OMP45-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// OMP45-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// OMP45:       [[FAIL]]
// OMP45:       call void [[HVT0:@.+]]([[S2]]* [[LOCAL_THIS1]])
// OMP45-NEXT:  br label %[[END]]
// OMP45:       [[END]]
// OMP45:       call void @__kmpc_end_critical(

// Check that the offloading functions are emitted and that the arguments are
// correct and loaded correctly for the target regions of the callees of bar().

// CHECK:       define internal void [[HVT7]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_THIS:%.+]] = alloca [[S1]]*
// CHECK:       [[LOCAL_B:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_C:%.+]] = alloca i16*
// CHECK-DAG:   store [[S1]]* [[ARG_THIS:%.+]], [[S1]]** [[LOCAL_THIS]]
// CHECK-DAG:   store i[[SZ]] [[ARG_B:%.+]], i[[SZ]]* [[LOCAL_B]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA1:%.+]], i[[SZ]]* [[LOCAL_VLA1]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA2:%.+]], i[[SZ]]* [[LOCAL_VLA2]]
// CHECK-DAG:   store i16* [[ARG_C:%.+]], i16** [[LOCAL_C]]
// Store captures in the context.
// CHECK-DAG:   [[REF_THIS:%.+]] = load [[S1]]*, [[S1]]** [[LOCAL_THIS]],
// CHECK-64-DAG:[[REF_B:%.+]] = bitcast i[[SZ]]* [[LOCAL_B]] to i32*
// CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA1]],
// CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA2]],
// CHECK-DAG:   [[REF_C:%.+]] = load i16*, i16** [[LOCAL_C]],
// Use captures.
// CHECK-DAG:   getelementptr inbounds [[S1]], [[S1]]* [[REF_THIS]], i32 0, i32 0
// CHECK-64-DAG:load i32, i32* [[REF_B]]
// CHECK-32-DAG:load i32, i32* [[LOCAL_B]]
// CHECK-DAG:   getelementptr inbounds i16, i16* [[REF_C]], i[[SZ]] %{{.+}}


// CHECK:       define internal void [[HVT6]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AAA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca [10 x i32]*
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AA:%.+]], i[[SZ]]* [[LOCAL_AA]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AAA:%.+]], i[[SZ]]* [[LOCAL_AAA]]
// CHECK-DAG:   store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
// Store captures in the context.
// CHECK-64-DAG:   [[REF_A:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
// CHECK-DAG:      [[REF_AA:%.+]] = bitcast i[[SZ]]* [[LOCAL_AA]] to i16*
// CHECK-DAG:      [[REF_AAA:%.+]] = bitcast i[[SZ]]* [[LOCAL_AAA]] to i8*
// CHECK-DAG:      [[REF_B:%.+]] = load [10 x i32]*, [10 x i32]** [[LOCAL_B]],
// Use captures.
// CHECK-64-DAG:   load i32, i32* [[REF_A]]
// CHECK-DAG:      load i16, i16* [[REF_AA]]
// CHECK-DAG:      load i8, i8* [[REF_AAA]]
// CHECK-32-DAG:   load i32, i32* [[LOCAL_A]]
// CHECK-DAG:      getelementptr inbounds [10 x i32], [10 x i32]* [[REF_B]], i[[SZ]] 0, i[[SZ]] 2

// CHECK:       define internal void [[HVT5]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca [10 x i32]*
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AA:%.+]], i[[SZ]]* [[LOCAL_AA]]
// CHECK-DAG:   store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
// Store captures in the context.
// CHECK-64-DAG:[[REF_A:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
// CHECK-DAG:   [[REF_AA:%.+]] = bitcast i[[SZ]]* [[LOCAL_AA]] to i16*
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x i32]*, [10 x i32]** [[LOCAL_B]],
// Use captures.
// CHECK-64-DAG:   load i32, i32* [[REF_A]]
// CHECK-32-DAG:   load i32, i32* [[LOCAL_A]]
// CHECK-DAG:   load i16, i16* [[REF_AA]]
// CHECK-DAG:   getelementptr inbounds [10 x i32], [10 x i32]* [[REF_B]], i[[SZ]] 0, i[[SZ]] 2

// OMP50: define internal void @__omp_offloading_{{.+}}_{{.+}}bar{{.+}}_l{{[0-9]+}}(i[[SZ]] noundef %{{.+}})

// OMP50: define {{.*}}@{{.*}}zee{{.*}}

// OMP50:       [[LOCAL_THIS:%.+]] = alloca [[S2]]*
// OMP50:       [[BP:%.+]] = alloca [1 x i8*]
// OMP50:       [[P:%.+]] = alloca [1 x i8*]
// OMP50:       [[LOCAL_THIS1:%.+]] = load [[S2]]*, [[S2]]** [[LOCAL_THIS]]
// OMP50:       [[ARR_IDX:%.+]] = getelementptr inbounds [[S2]], [[S2]]* [[LOCAL_THIS1]], i[[SZ]] 0
// OMP50:       [[ARR_IDX2:%.+]] = getelementptr inbounds [[S2]], [[S2]]* [[LOCAL_THIS1]], i[[SZ]] 0

// OMP50-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
// OMP50-DAG:   [[PADDR0:%.+]] =  getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
// OMP50-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to [[S2]]**
// OMP50-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to [[S2]]**
// OMP50-DAG:   store [[S2]]* [[ARR_IDX]], [[S2]]** [[CBPADDR0]]
// OMP50-DAG:   store [[S2]]* [[ARR_IDX2]], [[S2]]** [[CPADDR0]]

// OMP50:       [[BPR:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
// OMP50:       [[PR:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
// OMP50:       [[RET:%.+]] = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 1, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[SIZET9]], i32 0, i32 0), i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[MAPT10]], i32 0, i32 0), i8** null, i8** null)
// OMP50-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// OMP50-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// OMP50:       [[FAIL]]
// OMP50:       call void [[HVT0:@.+]]([[S2]]* [[LOCAL_THIS1]])
// OMP50-NEXT:  br label %[[END]]
// OMP50:       [[END]]

void bar () {
#define pragma_target _Pragma("omp target")
pragma_target
{
  global = 0;
#pragma omp parallel shared(global)
  global = 1;
}
}

class S2 {
  int a, b, c;

public:
  void zee() {
#pragma omp critical
    #pragma omp target map(this[0])
      a++;
  }
};

// CHECK:     define internal void @.omp_offloading.requires_reg()
// CHECK:     call void @__tgt_register_requires(i64 1)
// CHECK:     ret void

int main () {
  S2 bar;
  bar.zee();
}

#endif
