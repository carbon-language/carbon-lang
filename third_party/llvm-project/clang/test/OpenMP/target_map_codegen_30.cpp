// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32

// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32

// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32

// RUN: %clang_cc1 -DCK31 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK31

// CK31-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK31: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK31: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 1059]

// CK31-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK31: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK31: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 1063]

// CK31-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (int ii){
  // Map of a scalar.
  int a = ii;

  // Close.
  // Region 00
  // CK31-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null, i8** null)
  // CK31-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK31-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK31-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK31-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK31-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK31-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK31-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK31-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK31: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target map(close, tofrom: a)
  {
   a++;
  }

  // Always Close.
  // Region 01
  // CK31-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null, i8** null)
  // CK31-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK31-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK31-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK31-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK31-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK31-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK31-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK31-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK31: call void [[CALL01:@.+]](i32* {{[^,]+}})
  #pragma omp target map(always close tofrom: a)
  {
   a++;
  }
}
// CK31: define {{.+}}[[CALL00]]
// CK31: define {{.+}}[[CALL01]]

#endif // CK31
#endif
