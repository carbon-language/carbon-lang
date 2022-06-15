// Test host codegen.
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#pragma omp declare target
typedef struct {
  int *arr;
} MyObject;

MyObject *objects;
#pragma omp end declare target

// CHECK-DAG: [[SIZES0:@.+]] = private unnamed_addr constant [1 x i64] [i64 {{8|4}}]
// CHECK-DAG: [[MAPS0:@.+]] = private unnamed_addr constant [1 x i64] [i64 17]
// CHECK-DAG: [[SIZES1:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 4]
// CHECK-DAG: [[MAPS1:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 281474976710673]
// CHECK: @main
int main(void) {
// CHECK: [[BPTR0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* %{{.+}}, i32 0, i32 0
// CHECK: [[BPTRC0:%.+]] = bitcast i8** [[BPTR0]] to %struct.MyObject***
// CHECK: store %struct.MyObject** @objects, %struct.MyObject*** [[BPTRC0]],
// CHECK: call void @__tgt_target_data_begin_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[SIZES0]], i32 0, i32 0), i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[MAPS0]], i32 0, i32 0), i8** null, i8** null)
#pragma omp target enter data map(to : objects [1:1])
// CHECK: [[OBJ:%.+]] = load %struct.MyObject*, %struct.MyObject** @objects,
// CHECK: [[BPTR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* %{{.+}}, i32 0, i32 0
// CHECK: [[BPTRC0:%.+]] = bitcast i8** [[BPTR0]] to %struct.MyObject**
// CHECK: store %struct.MyObject* [[OBJ]], %struct.MyObject** [[BPTRC0]],
// CHECK: call void @__tgt_target_data_begin_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** %{{.+}}, i8** %{{.+}}, i64* %{{.+}}, i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[MAPS1]], i32 0, i32 0), i8** null, i8** null)
#pragma omp target enter data map(to : objects[1].arr [0:1])

  return 0;
}
#endif
