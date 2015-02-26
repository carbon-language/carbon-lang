// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -g -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
//
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-LABEL: define {{.*void}} @{{.*}}simple{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void simple(float *a, float *b, float *c, float *d) {
  #pragma omp simd
// CHECK: store i32 0, i32* [[OMP_IV:%[^,]+]]

// CHECK: [[IV:%.+]] = load i32* [[OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP1_ID:[0-9]+]]
// CHECK-NEXT: [[CMP:%.+]] = icmp slt i32 [[IV]], 6
// CHECK-NEXT: br i1 [[CMP]], label %[[SIMPLE_LOOP1_BODY:.+]], label %[[SIMPLE_LOOP1_END:[^,]+]]
  for (int i = 3; i < 32; i += 5) {
// CHECK: [[SIMPLE_LOOP1_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i32* [[OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP1_ID]]
// CHECK: [[CALC_I_1:%.+]] = mul nsw i32 [[IV1_1]], 5
// CHECK-NEXT: [[CALC_I_2:%.+]] = add nsw i32 3, [[CALC_I_1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP1_ID]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP1_ID]]
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32* [[OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP1_ID]]
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP1_ID]]
// br label %{{.+}}, !llvm.loop !{{.+}}
  }
// CHECK: [[SIMPLE_LOOP1_END]]

  #pragma omp simd
// CHECK: store i32 0, i32* [[OMP_IV2:%[^,]+]]

// CHECK: [[IV2:%.+]] = load i32* [[OMP_IV2]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID:[0-9]+]]
// CHECK-NEXT: [[CMP2:%.+]] = icmp slt i32 [[IV2]], 9
// CHECK-NEXT: br i1 [[CMP2]], label %[[SIMPLE_LOOP2_BODY:.+]], label %[[SIMPLE_LOOP2_END:[^,]+]]
  for (int i = 10; i > 1; i--) {
// CHECK: [[SIMPLE_LOOP2_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV2_0:%.+]] = load i32* [[OMP_IV2]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
// FIXME: It is interesting, why the following "mul 1" was not constant folded?
// CHECK-NEXT: [[IV2_1:%.+]] = mul nsw i32 [[IV2_0]], 1
// CHECK-NEXT: [[LC_I_1:%.+]] = sub nsw i32 10, [[IV2_1]]
// CHECK-NEXT: store i32 [[LC_I_1]], i32* {{.+}}, !llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
    a[i]++;
// CHECK: [[IV2_2:%.+]] = load i32* [[OMP_IV2]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
// CHECK-NEXT: [[ADD2_2:%.+]] = add nsw i32 [[IV2_2]], 1
// CHECK-NEXT: store i32 [[ADD2_2]], i32* [[OMP_IV2]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
// br label {{.+}}, !llvm.loop ![[SIMPLE_LOOP2_ID]]
  }
// CHECK: [[SIMPLE_LOOP2_END]]

  #pragma omp simd
// CHECK: store i64 0, i64* [[OMP_IV3:%[^,]+]]

// CHECK: [[IV3:%.+]] = load i64* [[OMP_IV3]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP3_ID:[0-9]+]]
// CHECK-NEXT: [[CMP3:%.+]] = icmp ult i64 [[IV3]], 4
// CHECK-NEXT: br i1 [[CMP3]], label %[[SIMPLE_LOOP3_BODY:.+]], label %[[SIMPLE_LOOP3_END:[^,]+]]
  for (unsigned long long it = 2000; it >= 600; it-=400) {
// CHECK: [[SIMPLE_LOOP3_BODY]]
// Start of body: calculate it from IV:
// CHECK: [[IV3_0:%.+]] = load i64* [[OMP_IV3]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP3_ID]]
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul i64 [[IV3_0]], 400
// CHECK-NEXT: [[LC_IT_2:%.+]] = sub i64 2000, [[LC_IT_1]]
// CHECK-NEXT: store i64 [[LC_IT_2]], i64* {{.+}}, !llvm.mem.parallel_loop_access ![[SIMPLE_LOOP3_ID]]
    a[it]++;
// CHECK: [[IV3_2:%.+]] = load i64* [[OMP_IV3]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP3_ID]]
// CHECK-NEXT: [[ADD3_2:%.+]] = add i64 [[IV3_2]], 1
// CHECK-NEXT: store i64 [[ADD3_2]], i64* [[OMP_IV3]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP3_ID]]
  }
// CHECK: [[SIMPLE_LOOP3_END]]

  #pragma omp simd
// CHECK: store i32 0, i32* [[OMP_IV4:%[^,]+]]

// CHECK: [[IV4:%.+]] = load i32* [[OMP_IV4]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP4_ID:[0-9]+]]
// CHECK-NEXT: [[CMP4:%.+]] = icmp slt i32 [[IV4]], 4
// CHECK-NEXT: br i1 [[CMP4]], label %[[SIMPLE_LOOP4_BODY:.+]], label %[[SIMPLE_LOOP4_END:[^,]+]]
  for (short it = 6; it <= 20; it-=-4) {
// CHECK: [[SIMPLE_LOOP4_BODY]]
// Start of body: calculate it from IV:
// CHECK: [[IV4_0:%.+]] = load i32* [[OMP_IV4]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP4_ID]]
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul nsw i32 [[IV4_0]], 4
// CHECK-NEXT: [[LC_IT_2:%.+]] = add nsw i32 6, [[LC_IT_1]]
// CHECK-NEXT: [[LC_IT_3:%.+]] = trunc i32 [[LC_IT_2]] to i16
// CHECK-NEXT: store i16 [[LC_IT_3]], i16* {{.+}}, !llvm.mem.parallel_loop_access ![[SIMPLE_LOOP4_ID]]

// CHECK: [[IV4_2:%.+]] = load i32* [[OMP_IV4]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP4_ID]]
// CHECK-NEXT: [[ADD4_2:%.+]] = add nsw i32 [[IV4_2]], 1
// CHECK-NEXT: store i32 [[ADD4_2]], i32* [[OMP_IV4]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP4_ID]]
  }
// CHECK: [[SIMPLE_LOOP4_END]]

  #pragma omp simd
// CHECK: store i32 0, i32* [[OMP_IV5:%[^,]+]]

// CHECK: [[IV5:%.+]] = load i32* [[OMP_IV5]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP5_ID:[0-9]+]]
// CHECK-NEXT: [[CMP5:%.+]] = icmp slt i32 [[IV5]], 26
// CHECK-NEXT: br i1 [[CMP5]], label %[[SIMPLE_LOOP5_BODY:.+]], label %[[SIMPLE_LOOP5_END:[^,]+]]
  for (unsigned char it = 'z'; it >= 'a'; it+=-1) {
// CHECK: [[SIMPLE_LOOP5_BODY]]
// Start of body: calculate it from IV:
// CHECK: [[IV5_0:%.+]] = load i32* [[OMP_IV5]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP5_ID]]
// CHECK-NEXT: [[IV5_1:%.+]] = mul nsw i32 [[IV5_0]], 1
// CHECK-NEXT: [[LC_IT_1:%.+]] = sub nsw i32 122, [[IV5_1]]
// CHECK-NEXT: [[LC_IT_2:%.+]] = trunc i32 [[LC_IT_1]] to i8
// CHECK-NEXT: store i8 [[LC_IT_2]], i8* {{.+}}, !llvm.mem.parallel_loop_access ![[SIMPLE_LOOP5_ID]]

// CHECK: [[IV5_2:%.+]] = load i32* [[OMP_IV5]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP5_ID]]
// CHECK-NEXT: [[ADD5_2:%.+]] = add nsw i32 [[IV5_2]], 1
// CHECK-NEXT: store i32 [[ADD5_2]], i32* [[OMP_IV5]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP5_ID]]
  }
// CHECK: [[SIMPLE_LOOP5_END]]

  #pragma omp simd
// FIXME: I think we would get wrong result using 'unsigned' in the loop below.
// So we'll need to add zero trip test for 'unsigned' counters.
//
// CHECK: store i32 0, i32* [[OMP_IV6:%[^,]+]]

// CHECK: [[IV6:%.+]] = load i32* [[OMP_IV6]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP6_ID:[0-9]+]]
// CHECK-NEXT: [[CMP6:%.+]] = icmp slt i32 [[IV6]], -8
// CHECK-NEXT: br i1 [[CMP6]], label %[[SIMPLE_LOOP6_BODY:.+]], label %[[SIMPLE_LOOP6_END:[^,]+]]
  for (int i=100; i<10; i+=10) {
// CHECK: [[SIMPLE_LOOP6_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV6_0:%.+]] = load i32* [[OMP_IV6]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP6_ID]]
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul nsw i32 [[IV6_0]], 10
// CHECK-NEXT: [[LC_IT_2:%.+]] = add nsw i32 100, [[LC_IT_1]]
// CHECK-NEXT: store i32 [[LC_IT_2]], i32* {{.+}}, !llvm.mem.parallel_loop_access ![[SIMPLE_LOOP6_ID]]

// CHECK: [[IV6_2:%.+]] = load i32* [[OMP_IV6]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP6_ID]]
// CHECK-NEXT: [[ADD6_2:%.+]] = add nsw i32 [[IV6_2]], 1
// CHECK-NEXT: store i32 [[ADD6_2]], i32* [[OMP_IV6]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP6_ID]]
  }
// CHECK: [[SIMPLE_LOOP6_END]]

  int A;
  #pragma omp simd lastprivate(A)
// Clause 'lastprivate' implementation is not completed yet.
// Test checks that one iteration is separated in presence of lastprivate.
//
// CHECK: store i64 0, i64* [[OMP_IV7:%[^,]+]]
// CHECK: br i1 true, label %[[SIMPLE_IF7_THEN:.+]], label %[[SIMPLE_IF7_END:[^,]+]]
// CHECK: [[SIMPLE_IF7_THEN]]
// CHECK: br label %[[SIMD_LOOP7_COND:[^,]+]]
// CHECK: [[SIMD_LOOP7_COND]]
// CHECK-NEXT: [[IV7:%.+]] = load i64* [[OMP_IV7]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP7_ID:[0-9]+]]
// CHECK-NEXT: [[CMP7:%.+]] = icmp slt i64 [[IV7]], 6
// CHECK-NEXT: br i1 [[CMP7]], label %[[SIMPLE_LOOP7_BODY:.+]], label %[[SIMPLE_LOOP7_END:[^,]+]]
  for (long long i = -10; i < 10; i += 3) {
// CHECK: [[SIMPLE_LOOP7_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV7_0:%.+]] = load i64* [[OMP_IV7]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP7_ID]]
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul nsw i64 [[IV7_0]], 3
// CHECK-NEXT: [[LC_IT_2:%.+]] = add nsw i64 -10, [[LC_IT_1]]
// CHECK-NEXT: store i64 [[LC_IT_2]], i64* {{.+}}, !llvm.mem.parallel_loop_access ![[SIMPLE_LOOP7_ID]]
    A = i;
// CHECK: [[IV7_2:%.+]] = load i64* [[OMP_IV7]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP7_ID]]
// CHECK-NEXT: [[ADD7_2:%.+]] = add nsw i64 [[IV7_2]], 1
// CHECK-NEXT: store i64 [[ADD7_2]], i64* [[OMP_IV7]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP7_ID]]
  }
// CHECK: [[SIMPLE_LOOP7_END]]
// Separated last iteration.
// CHECK: [[IV7_4:%.+]] = load i64* [[OMP_IV7]]
// CHECK-NEXT: [[LC_FIN_1:%.+]] = mul nsw i64 [[IV7_4]], 3
// CHECK-NEXT: [[LC_FIN_2:%.+]] = add nsw i64 -10, [[LC_FIN_1]]
// CHECK-NEXT: store i64 [[LC_FIN_2]], i64* [[ADDR_I:%[^,]+]]
// CHECK: [[LOAD_I:%.+]] = load i64* [[ADDR_I]]
// CHECK-NEXT: [[CONV_I:%.+]] = trunc i64 [[LOAD_I]] to i32
//
// CHECK: br label %[[SIMPLE_IF7_END]]
// CHECK: [[SIMPLE_IF7_END]]
//

// CHECK: ret void
}

template <class T, unsigned K> T tfoo(T a) { return a + K; }

template <typename T, unsigned N>
int templ1(T a, T *z) {
  #pragma omp simd collapse(N)
  for (int i = 0; i < N * 2; i++) {
    for (long long j = 0; j < (N + N + N + N); j += 2) {
      z[i + j] = a + tfoo<T, N>(i + j);
    }
  }
  return 0;
}

// Instatiation templ1<float,2>
// CHECK-LABEL: define {{.*i32}} @{{.*}}templ1{{.*}}(float {{.+}}, float* {{.+}})
// CHECK: store i64 0, i64* [[T1_OMP_IV:[^,]+]]
// ...
// CHECK: [[IV:%.+]] = load i64* [[T1_OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[T1_ID:[0-9]+]]
// CHECK-NEXT: [[CMP1:%.+]] = icmp slt i64 [[IV]], 16
// CHECK-NEXT: br i1 [[CMP1]], label %[[T1_BODY:.+]], label %[[T1_END:[^,]+]]
// CHECK: [[T1_BODY]]
// Loop counters i and j updates:
// CHECK: [[IV1:%.+]] = load i64* [[T1_OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[T1_ID]]
// CHECK-NEXT: [[I_1:%.+]] = sdiv i64 [[IV1]], 4
// CHECK-NEXT: [[I_1_MUL1:%.+]] = mul nsw i64 [[I_1]], 1
// CHECK-NEXT: [[I_1_ADD0:%.+]] = add nsw i64 0, [[I_1_MUL1]]
// CHECK-NEXT: [[I_2:%.+]] = trunc i64 [[I_1_ADD0]] to i32
// CHECK-NEXT: store i32 [[I_2]], i32* {{%.+}}{{.*}}!llvm.mem.parallel_loop_access ![[T1_ID]]
// CHECK: [[IV2:%.+]] = load i64* [[T1_OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[T1_ID]]
// CHECK-NEXT: [[J_1:%.+]] = srem i64 [[IV2]], 4
// CHECK-NEXT: [[J_2:%.+]] = mul nsw i64 [[J_1]], 2
// CHECK-NEXT: [[J_2_ADD0:%.+]] = add nsw i64 0, [[J_2]]
// CHECK-NEXT: store i64 [[J_2_ADD0]], i64* {{%.+}}{{.*}}!llvm.mem.parallel_loop_access ![[T1_ID]]
// simd.for.inc:
// CHECK: [[IV3:%.+]] = load i64* [[T1_OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[T1_ID]]
// CHECK-NEXT: [[INC:%.+]] = add nsw i64 [[IV3]], 1
// CHECK-NEXT: store i64 [[INC]], i64* [[T1_OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[T1_ID]]
// CHECK-NEXT: br label {{%.+}}
// CHECK: [[T1_END]]
// CHECK: ret i32 0
//
void inst_templ1() {
  float a;
  float z[100];
  templ1<float,2> (a, z);
}


typedef int MyIdx;

class IterDouble {
  double *Ptr;
public:
  IterDouble operator++ () const {
    IterDouble n;
    n.Ptr = Ptr + 1;
    return n;
  }
  bool operator < (const IterDouble &that) const {
    return Ptr < that.Ptr;
  }
  double & operator *() const {
    return *Ptr;
  }
  MyIdx operator - (const IterDouble &that) const {
    return (MyIdx) (Ptr - that.Ptr);
  }
  IterDouble operator + (int Delta) {
    IterDouble re;
    re.Ptr = Ptr + Delta;
    return re;
  }

  ///~IterDouble() {}
};

// CHECK-LABEL: define {{.*void}} @{{.*}}iter_simple{{.*}}
void iter_simple(IterDouble ia, IterDouble ib, IterDouble ic) {
//
// CHECK: store i32 0, i32* [[IT_OMP_IV:%[^,]+]]
// Calculate number of iterations before the loop body.
// CHECK: [[DIFF1:%.+]] = call {{.*}}i32 @{{.*}}IterDouble{{.*}}
// CHECK-NEXT: [[DIFF2:%.+]] = sub nsw i32 [[DIFF1]], 1
// CHECK-NEXT: [[DIFF3:%.+]] = add nsw i32 [[DIFF2]], 1
// CHECK-NEXT: [[DIFF4:%.+]] = sdiv i32 [[DIFF3]], 1
// CHECK-NEXT: [[DIFF5:%.+]] = sub nsw i32 [[DIFF4]], 1
// CHECK-NEXT: store i32 [[DIFF5]], i32* [[OMP_LAST_IT:%[^,]+]]{{.+}}
  #pragma omp simd

// CHECK: [[IV:%.+]] = load i32* [[IT_OMP_IV]]{{.+}} !llvm.mem.parallel_loop_access ![[ITER_LOOP_ID:[0-9]+]]
// CHECK-NEXT: [[LAST_IT:%.+]] = load i32* [[OMP_LAST_IT]]{{.+}}!llvm.mem.parallel_loop_access ![[ITER_LOOP_ID]]
// CHECK-NEXT: [[NUM_IT:%.+]] = add nsw i32 [[LAST_IT]], 1
// CHECK-NEXT: [[CMP:%.+]] = icmp slt i32 [[IV]], [[NUM_IT]]
// CHECK-NEXT: br i1 [[CMP]], label %[[IT_BODY:[^,]+]], label %[[IT_END:[^,]+]]
  for (IterDouble i = ia; i < ib; ++i) {
// CHECK: [[IT_BODY]]
// Start of body: calculate i from index:
// CHECK: [[IV1:%.+]] = load i32* [[IT_OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[ITER_LOOP_ID]]
// Call of operator+ (i, IV).
// CHECK: {{%.+}} = call {{.+}} @{{.*}}IterDouble{{.*}}!llvm.mem.parallel_loop_access ![[ITER_LOOP_ID]]
// ... loop body ...
   *i = *ic * 0.5;
// Float multiply and save result.
// CHECK: [[MULR:%.+]] = fmul double {{%.+}}, 5.000000e-01
// CHECK-NEXT: call {{.+}} @{{.*}}IterDouble{{.*}}
// CHECK: store double [[MULR:%.+]], double* [[RESULT_ADDR:%.+]], !llvm.mem.parallel_loop_access ![[ITER_LOOP_ID]]
   ++ic;
//
// CHECK: [[IV2:%.+]] = load i32* [[IT_OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[ITER_LOOP_ID]]
// CHECK-NEXT: [[ADD2:%.+]] = add nsw i32 [[IV2]], 1
// CHECK-NEXT: store i32 [[ADD2]], i32* [[IT_OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[ITER_LOOP_ID]]
// br label %{{.*}}, !llvm.loop ![[ITER_LOOP_ID]]
  }
// CHECK: [[IT_END]]
// CHECK: ret void
}


// CHECK-LABEL: define {{.*void}} @{{.*}}collapsed{{.*}}
void collapsed(float *a, float *b, float *c, float *d) {
  int i; // outer loop counter
  unsigned j; // middle loop couter, leads to unsigned icmp in loop header.
  // k declared in the loop init below
  short l; // inner loop counter
// CHECK: store i32 0, i32* [[OMP_IV:[^,]+]]
//
  #pragma omp simd collapse(4)

// CHECK: [[IV:%.+]] = load i32* [[OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[COLL1_LOOP_ID:[0-9]+]]
// CHECK-NEXT: [[CMP:%.+]] = icmp ult i32 [[IV]], 120
// CHECK-NEXT: br i1 [[CMP]], label %[[COLL1_BODY:[^,]+]], label %[[COLL1_END:[^,]+]]
  for (i = 1; i < 3; i++) // 2 iterations
    for (j = 2u; j < 5u; j++) //3 iterations
      for (int k = 3; k <= 6; k++) // 4 iterations
        for (l = 4; l < 9; ++l) // 5 iterations
        {
// CHECK: [[COLL1_BODY]]
// Start of body: calculate i from index:
// CHECK: [[IV1:%.+]] = load i32* [[OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[COLL1_LOOP_ID]]
// Calculation of the loop counters values.
// CHECK: [[CALC_I_1:%.+]] = udiv i32 [[IV1]], 60
// CHECK-NEXT: [[CALC_I_1_MUL1:%.+]] = mul i32 [[CALC_I_1]], 1
// CHECK-NEXT: [[CALC_I_2:%.+]] = add i32 1, [[CALC_I_1_MUL1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]
// CHECK: [[IV1_2:%.+]] = load i32* [[OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[COLL1_LOOP_ID]]
// CHECK-NEXT: [[CALC_J_1:%.+]] = udiv i32 [[IV1_2]], 20
// CHECK-NEXT: [[CALC_J_2:%.+]] = urem i32 [[CALC_J_1]], 3
// CHECK-NEXT: [[CALC_J_2_MUL1:%.+]] = mul i32 [[CALC_J_2]], 1
// CHECK-NEXT: [[CALC_J_3:%.+]] = add i32 2, [[CALC_J_2_MUL1]]
// CHECK-NEXT: store i32 [[CALC_J_3]], i32* [[LC_J:.+]]
// CHECK: [[IV1_3:%.+]] = load i32* [[OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[COLL1_LOOP_ID]]
// CHECK-NEXT: [[CALC_K_1:%.+]] = udiv i32 [[IV1_3]], 5
// CHECK-NEXT: [[CALC_K_2:%.+]] = urem i32 [[CALC_K_1]], 4
// CHECK-NEXT: [[CALC_K_2_MUL1:%.+]] = mul i32 [[CALC_K_2]], 1
// CHECK-NEXT: [[CALC_K_3:%.+]] = add i32 3, [[CALC_K_2_MUL1]]
// CHECK-NEXT: store i32 [[CALC_K_3]], i32* [[LC_K:.+]]
// CHECK: [[IV1_4:%.+]] = load i32* [[OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[COLL1_LOOP_ID]]
// CHECK-NEXT: [[CALC_L_1:%.+]] = urem i32 [[IV1_4]], 5
// CHECK-NEXT: [[CALC_L_1_MUL1:%.+]] = mul i32 [[CALC_L_1]], 1
// CHECK-NEXT: [[CALC_L_2:%.+]] = add i32 4, [[CALC_L_1_MUL1]]
// CHECK-NEXT: [[CALC_L_3:%.+]] = trunc i32 [[CALC_L_2]] to i16
// CHECK-NEXT: store i16 [[CALC_L_3]], i16* [[LC_L:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* [[RESULT_ADDR:%.+]]{{.+}}!llvm.mem.parallel_loop_access ![[COLL1_LOOP_ID]]
    float res = b[j] * c[k];
    a[i] = res * d[l];
// CHECK: [[IV2:%.+]] = load i32* [[OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[COLL1_LOOP_ID]]
// CHECK-NEXT: [[ADD2:%.+]] = add i32 [[IV2]], 1
// CHECK-NEXT: store i32 [[ADD2]], i32* [[OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[COLL1_LOOP_ID]]
// br label %{{[^,]+}}, !llvm.loop ![[COLL1_LOOP_ID]]
// CHECK: [[COLL1_END]]
  }
// i,j,l are updated; k is not updated.
// CHECK: store i32 3, i32* [[I:%[^,]+]]
// CHECK-NEXT: store i32 5, i32* [[I:%[^,]+]]
// CHECK-NEXT: store i16 9, i16* [[I:%[^,]+]]
// CHECK: ret void
}

extern char foo();

// CHECK-LABEL: define {{.*void}} @{{.*}}widened{{.*}}
void widened(float *a, float *b, float *c, float *d) {
  int i; // outer loop counter
  short j; // inner loop counter
// Counter is widened to 64 bits.
// CHECK: store i64 0, i64* [[OMP_IV:[^,]+]]
//
  #pragma omp simd collapse(2)

// CHECK: [[IV:%.+]] = load i64* [[OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[WIDE1_LOOP_ID:[0-9]+]]
// CHECK-NEXT: [[LI:%.+]] = load i64* [[OMP_LI:%[^,]+]]{{.+}}!llvm.mem.parallel_loop_access ![[WIDE1_LOOP_ID]]
// CHECK-NEXT: [[NUMIT:%.+]] = add nsw i64 [[LI]], 1
// CHECK-NEXT: [[CMP:%.+]] = icmp slt i64 [[IV]], [[NUMIT]]
// CHECK-NEXT: br i1 [[CMP]], label %[[WIDE1_BODY:[^,]+]], label %[[WIDE1_END:[^,]+]]
  for (i = 1; i < 3; i++) // 2 iterations
    for (j = 0; j < foo(); j++) // foo() iterations
  {
// CHECK: [[WIDE1_BODY]]
// Start of body: calculate i from index:
// CHECK: [[IV1:%.+]] = load i64* [[OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[WIDE1_LOOP_ID]]
// Calculation of the loop counters values...
// CHECK: store i32 {{[^,]+}}, i32* [[LC_I:.+]]
// CHECK: [[IV1_2:%.+]] = load i64* [[OMP_IV]]{{.+}}!llvm.mem.parallel_loop_access ![[WIDE1_LOOP_ID]]
// CHECK: store i16 {{[^,]+}}, i16* [[LC_J:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* [[RESULT_ADDR:%.+]]{{.+}}!llvm.mem.parallel_loop_access ![[WIDE1_LOOP_ID]]
    float res = b[j] * c[j];
    a[i] = res * d[i];
// CHECK: [[IV2:%.+]] = load i64* [[OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[WIDE1_LOOP_ID]]
// CHECK-NEXT: [[ADD2:%.+]] = add nsw i64 [[IV2]], 1
// CHECK-NEXT: store i64 [[ADD2]], i64* [[OMP_IV]]{{.*}}!llvm.mem.parallel_loop_access ![[WIDE1_LOOP_ID]]
// br label %{{[^,]+}}, !llvm.loop ![[WIDE1_LOOP_ID]]
// CHECK: [[WIDE1_END]]
  }
// i,j are updated.
// CHECK: store i32 3, i32* [[I:%[^,]+]]
// CHECK: store i16
// CHECK: ret void
}

void parallel_simd(float *a) {
#pragma omp parallel
#pragma omp simd
  // CHECK-NOT: __kmpc_global_thread_num
  for (unsigned i = 131071; i <= 2147483647; i += 127)
    a[i] += i;
}

#endif // HEADER

