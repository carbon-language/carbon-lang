// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s | FileCheck %s

// CHECK: define linkonce_odr void @_Z11inline_funci
inline void inline_func(int n) {
  // CHECK: call i32 @_ZZ11inline_funciENKUlvE_clEv
  int i = []{ return 1; }();

  // CHECK: call i32 @_ZZ11inline_funciENKUlvE0_clEv
  int j = [=] { return n + i; }();

  // CHECK: call double @_ZZ11inline_funciENKUlvE1_clEv
  int k = [=] () -> double { return n + i; }();

  // CHECK: call i32 @_ZZ11inline_funciENKUliE_clEi
  int l = [=] (int x) -> int { return x + i; }(n);

  int inner(int i = []{ return 1; }());
  // CHECK: call i32 @_ZZ11inline_funciENKUlvE2_clEv
  // CHECK-NEXT: call i32 @_Z5inneri
  inner();

  // CHECK-NEXT: ret void
}

void call_inline_func() {
  inline_func(17);
}

struct S {
  void f(int = []{return 1;}()
             + []{return 2;}(),
         int = []{return 3;}());
  void g(int, int);
};

void S::g(int i = []{return 1;}(),
          int j = []{return 2; }()) {}

// CHECK: define void @_Z6test_S1S
void test_S(S s) {
  // CHECK: call i32 @_ZZN1S1fEiiEd0_NKUlvE_clEv
  // CHECK-NEXT: call i32 @_ZZN1S1fEiiEd0_NKUlvE0_clEv
  // CHECK-NEXT: add nsw i32
  // CHECK-NEXT: call i32 @_ZZN1S1fEiiEd_NKUlvE_clEv
  // CHECK-NEXT: call void @_ZN1S1fEii
  s.f();

  // NOTE: These manglings don't actually matter that much, because
  // the lambdas in the default arguments of g() won't be seen by
  // multiple translation units. We check them mainly to ensure that they don't 
  // get the special mangling for lambdas in in-class default arguments.
  // CHECK: call i32 @_ZNK1SUlvE_clEv
  // CHECK-NEXT: call i32 @_ZNK1SUlvE0_clEv
  // CHECK-NEXT: call void @_ZN1S1gEi
  s.g();

  // CHECK-NEXT: ret void
}

template<typename T>
struct ST {
  void f(T = []{return T() + 1;}()
           + []{return T() + 2;}(),
         T = []{return T(3);}());
};

// CHECK: define void @_Z7test_ST2STIdE
void test_ST(ST<double> st) {
  // CHECK: call double @_ZZN2ST1fET_S0_Ed0_NKUlvE_clEv
  // CHECK-NEXT: call double @_ZZN2ST1fET_S0_Ed0_NKUlvE0_clEv
  // CHECK-NEXT: fadd double
  // CHECK-NEXT: call double @_ZZN2ST1fET_S0_Ed_NKUlvE_clEv
  // CHECK-NEXT: call void @_ZN2STIdE1fEdd
  st.f();

  // CHECK-NEXT: ret void
}

template<typename T> 
struct StaticMembers {
  static T x;
  static T y;
  static T z;
};

template<typename T> int accept_lambda(T);

template<typename T>
T StaticMembers<T>::x = []{return 1;}() + []{return 2;}();

template<typename T>
T StaticMembers<T>::y = []{return 3;}();

template<typename T>
T StaticMembers<T>::z = accept_lambda([]{return 4;});

// CHECK: define internal void @__cxx_global_var_init()
// CHECK: call i32 @_ZNK13StaticMembersIfE1xMUlvE_clEv
// CHECK-NEXT: call i32 @_ZNK13StaticMembersIfE1xMUlvE0_clEv
// CHECK-NEXT: add nsw
template float StaticMembers<float>::x;

// CHECK: define internal void @__cxx_global_var_init1()
// CHECK: call i32 @_ZNK13StaticMembersIfE1yMUlvE_clEv
template float StaticMembers<float>::y;

// CHECK: define internal void @__cxx_global_var_init2()
// CHECK: call i32 @_Z13accept_lambdaIN13StaticMembersIfE1zMUlvE_EEiT_
template float StaticMembers<float>::z;

struct Members {
  int x = [] { return 1; }() + [] { return 2; }();
  int y = [] { return 3; }();
};

void test_Members() {
  // CHECK: define linkonce_odr void @_ZN7MembersC2Ev
  // CHECK: call i32 @_ZNK7Members1xMUlvE_clEv
  // CHECK-NEXT: call i32 @_ZNK7Members1xMUlvE0_clE
  // CHECK-NEXT: add nsw i32
  // CHECK: call i32 @_ZNK7Members1yMUlvE_clEv
  Members members;
  // CHECK: ret void
}
