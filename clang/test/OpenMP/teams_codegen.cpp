// expected-no-diagnostics
#ifndef HEADER
#define HEADER
// Test host codegen.
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -omptargets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -omptargets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -omptargets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -omptargets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -omptargets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple i386-unknown-unknown -omptargets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32
#ifdef CK1

int Gbla;
long long Gblb;
int &Gblc = Gbla;

// CK1-LABEL: teams_argument_global_local
int teams_argument_global_local(int a){
  int comp = 1;

  int la = 23;
  float lc = 25.0;

  // CK1: call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 1, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 0, i32 0)
  // CK1: call void @{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target
  #pragma omp teams
  {
    ++comp;
  }

  // CK1-DAG: call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 2, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 0)
  // CK1-DAG: [[NT]] = load i32, i32* [[NTA:%[^,]+]],

  // CK1: call void @{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target
  #pragma omp teams num_teams(la)
  {
    ++comp;
  }

  // CK1-DAG: call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 2, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 0, i32 [[NT:%[^,]+]])
  // CK1-DAG: [[NT]] = load i32, i32* [[NTA:%[^,]+]],

  // CK1: call void @{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target
  #pragma omp teams thread_limit(la)
  {
    ++comp;
  }

  // CK1-DAG: call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 5, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 [[TL:%[^,]+]])

  // CK1-DAG: [[NT]] = add nsw i32 [[NTA:%[^,]+]], [[NTB:%[^,]+]]
  // CK1-DAG: [[NTA]] = load i32, i32* @Gbla,
  // CK1-DAG: [[NTB]] = load i32, i32* %{{.+}},

  // CK1-DAG: [[TL]] = trunc i64 [[TLA:%[^,]+]] to i32
  // CK1-DAG: [[TLA]] = add nsw i64 [[TLB:%[^,]+]], [[TLC:%[^,]+]]
  // CK1-DAG: [[TLC]] = fptosi float [[TLD:%[^,]+]] to i64
  // CK1-DAG: [[TLD]] = load float, float* %{{.+}},
  // CK1-DAG: [[TLB]] = load i64, i64* @Gblb,

  // CK1: call void @{{.+}}(i{{.+}} {{.+}}, i{{.+}} {{.+}}, i{{.+}} {{.+}}, i{{.+}} {{.+}}, i{{.+}} {{.+}})
  #pragma omp target
  #pragma omp teams num_teams(Gbla+a) thread_limit(Gblb+(long long)lc)
  {
    ++comp;
  }

  // CK1-DAG: call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 {{.+}}, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 [[TL:%[^,]+]])

  // CK1-DAG: [[NT]] = add nsw i32 [[NTA:%[^,]+]], 1
  // CK1-DAG: [[NTA]] = load i32, i32* @Gbla,

  // CK1-DAG: [[TL]] = add nsw i32 [[TLA:%[^,]+]], 2
  // CK1-DAG: [[TLA]] = load i32, i32* @Gbla,

  // CK1: call void @{{.+}}(i{{.+}} {{.+}}
  #pragma omp target
  #pragma omp teams num_teams(Gblc+1) thread_limit(Gblc+2)
  {
    comp += Gblc;
  }

  return comp;
}

#endif // CK1

// Test host codegen.
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -omptargets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -omptargets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -omptargets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple i386-unknown-unknown -omptargets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -omptargets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple i386-unknown-unknown -omptargets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32
#ifdef CK2

// CK2-DAG: [[SSI:%.+]] = type { i32, float }
// CK2-DAG: [[SSL:%.+]] = type { i64, float }
template <typename T>
struct SS{
  T a;
  float b;
};

SS<int> Gbla;
SS<long long> Gblb;

// CK2-LABEL: teams_template_arg
int teams_template_arg(void) {
  int comp = 1;

  SS<int> la;
  SS<long long> lb;

  // CK2-DAG: call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 3, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 [[TL:%[^,]+]])

  // CK2-DAG: [[NT]] = load i32, i32* getelementptr inbounds ([[SSI]], [[SSI]]* @Gbla, i32 0, i32 0)

  // CK2-DAG: [[TL]] = trunc i64 [[TLA:%[^,]+]] to i32
  // CK2-DAG: [[TLA]] = fptosi float [[TLB:%[^,]+]] to i64
  // CK2-DAG: [[TLB]] = load float, float* [[TLC:%[^,]+]],
  // CK2-DAG: [[TLC]] = getelementptr inbounds [[SSI]], [[SSI]]* %{{.+}}, i32 0, i32 1

  // CK2: call void @{{.+}}({{.+}} {{.+}}, {{.+}} {{.+}}, {{.+}} {{.+}})
  #pragma omp target
  #pragma omp teams num_teams(Gbla.a) thread_limit((long long)la.b)
  {
    ++comp;
  }

  // CK2-DAG: call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 3, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 [[TL:%[^,]+]])

  // CK2-DAG: [[TL]] = trunc i64 [[TLD:%[^,]+]] to i32
  // CK2-DAG: [[TLD]] = load i64, i64* getelementptr inbounds ([[SSL]], [[SSL]]* @Gblb, i32 0, i32 0),

  // CK2-DAG: [[NT]] = trunc i64 [[NTA:%[^,]+]] to i32
  // CK2-DAG: [[NTA]] = fptosi float [[NTB:%[^,]+]] to i64
  // CK2-DAG: [[NTB]] = load float, float* [[NTC:%[^,]+]],
  // CK2-DAG: [[NTC]] = getelementptr inbounds [[SSL]], [[SSL]]* %{{.+}}, i32 0, i32 1

  // CK2: call void @{{.+}}({{.+}} {{.+}}, {{.+}} {{.+}}, {{.+}} {{.+}})
  #pragma omp target
  #pragma omp teams num_teams((long long)lb.b) thread_limit(Gblb.a)
  {
    ++comp;
  }
  return comp;
}
#endif // CK2

// Test host codegen.
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -omptargets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -omptargets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -omptargets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple i386-unknown-unknown -omptargets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -omptargets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -triple i386-unknown-unknown -omptargets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-32
#ifdef CK3

// CK3: [[SSI:%.+]] = type { i32, float }
// CK3-LABEL: teams_template_struct

template <typename T, int X, long long Y>
struct SS{
  T a;
  float b;

  int foo(void) {
    int comp = 1;

    // CK3-DAG: call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 2, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 123)

    // CK3-DAG: [[NT]] = load i32, i32* [[NTA:%[^,]+]],
    // CK3-DAG: [[NTA]] = getelementptr inbounds [[SSI]], [[SSI]]* [[NTB:%[^,]+]], i32 0, i32 0
    // CK3-DAG: [[NTB]] = load [[SSI]]*, [[SSI]]** %{{.+}},

    // CK3: call void @{{.+}}({{.+}} {{.+}}, {{.+}} {{.+}})
    #pragma omp target
    #pragma omp teams num_teams(a) thread_limit(X)
    {
      ++comp;
    }

    // CK3-DAG: call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 2, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 456, i32 [[TL:%[^,]+]])

    // CK3-DAG: [[TL]] = add nsw i32 [[TLA:%[^,]+]], 123
    // CK3-DAG: [[TLA]] = fptosi float [[TLB:%[^,]+]] to i32
    // CK3-DAG: [[TLB]] = load float, float* [[TLC:%[^,]+]],
    // CK3-DAG: [[TLC]] = getelementptr inbounds [[SSI]], [[SSI]]* [[THIS:%[^,]+]], i32 0, i32 1

    // CK3: call void @{{.+}}({{.+}} {{.+}}, {{.+}} {{.+}})
    #pragma omp target
    #pragma omp teams num_teams(Y) thread_limit((int)b+X)
    {
      ++comp;
    }
    return comp;
  }
};

int teams_template_struct(void) {
  SS<int, 123, 456> V;
  return V.foo();

}
#endif // CK3
#endif
