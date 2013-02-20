// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -verify -fexceptions -fcxx-exceptions -triple x86_64-linux-gnu | FileCheck %s

void h();

template<typename T> void f() noexcept(sizeof(T) == 4) { h(); }
template<typename T> void g() noexcept(sizeof(T) == 4);

template<typename T> struct S {
  static void f() noexcept(sizeof(T) == 4) { h(); }
  static void g() noexcept(sizeof(T) == 4);
};

// CHECK: define {{.*}} @_Z1fIsEvv() #0 {
template<> void f<short>() { h(); }
// CHECK: define {{.*}} @_Z1fIA2_sEvv() #1 {
template<> void f<short[2]>() noexcept { h(); }

// CHECK: define {{.*}} @_ZN1SIsE1fEv()
// CHECK-NOT: #1
template<> void S<short>::f() { h(); }
// CHECK: define {{.*}} @_ZN1SIA2_sE1fEv() #1
template<> void S<short[2]>::f() noexcept { h(); }

// CHECK: define {{.*}} @_Z1fIDsEvv() #0 {
template void f<char16_t>();
// CHECK: define {{.*}} @_Z1fIA2_DsEvv() #1  {
template void f<char16_t[2]>();

// CHECK: define {{.*}} @_ZN1SIDsE1fEv()
// CHECK-NOT: #1
template void S<char16_t>::f();
// CHECK: define {{.*}} @_ZN1SIA2_DsE1fEv() #1
template void S<char16_t[2]>::f();

void h() {
  // CHECK: define {{.*}} @_Z1fIiEvv() #1 {
  f<int>();
  // CHECK: define {{.*}} @_Z1fIA2_iEvv() #0 {
  f<int[2]>();

  // CHECK: define {{.*}} @_ZN1SIiE1fEv() #1
  S<int>::f();
  // CHECK: define {{.*}} @_ZN1SIA2_iE1fEv()
  // CHECK-NOT: #1
  S<int[2]>::f();

  // CHECK: define {{.*}} @_Z1fIfEvv() #1 {
  void (*f1)() = &f<float>;
  // CHECK: define {{.*}} @_Z1fIdEvv() #0 {
  void (*f2)() = &f<double>;

  // CHECK: define {{.*}} @_ZN1SIfE1fEv() #1
  void (*f3)() = &S<float>::f;
  // CHECK: define {{.*}} @_ZN1SIdE1fEv()
  // CHECK-NOT: #1
  void (*f4)() = &S<double>::f;

  // CHECK: define {{.*}} @_Z1fIA4_cEvv() #1 {
  (void)&f<char[4]>;
  // CHECK: define {{.*}} @_Z1fIcEvv() #0 {
  (void)&f<char>;

  // CHECK: define {{.*}} @_ZN1SIA4_cE1fEv() #1
  (void)&S<char[4]>::f;
  // CHECK: define {{.*}} @_ZN1SIcE1fEv()
  // CHECK-NOT: #1
  (void)&S<char>::f;
}

// CHECK: define {{.*}} @_Z1iv
void i() {
  // CHECK: declare {{.*}} @_Z1gIiEvv() #1
  g<int>();
  // CHECK: declare {{.*}} @_Z1gIA2_iEvv()
  // CHECK-NOT: #1
  g<int[2]>();

  // CHECK: declare {{.*}} @_ZN1SIiE1gEv() #1
  S<int>::g();
  // CHECK: declare {{.*}} @_ZN1SIA2_iE1gEv()
  // CHECK-NOT: #1
  S<int[2]>::g();

  // CHECK: declare {{.*}} @_Z1gIfEvv() #1
  void (*g1)() = &g<float>;
  // CHECK: declare {{.*}} @_Z1gIdEvv()
  // CHECK-NOT: #1
  void (*g2)() = &g<double>;

  // CHECK: declare {{.*}} @_ZN1SIfE1gEv() #1
  void (*g3)() = &S<float>::g;
  // CHECK: declare {{.*}} @_ZN1SIdE1gEv()
  // CHECK-NOT: #1
  void (*g4)() = &S<double>::g;

  // CHECK: declare {{.*}} @_Z1gIA4_cEvv() #1
  (void)&g<char[4]>;
  // CHECK: declare {{.*}} @_Z1gIcEvv()
  // CHECK-NOT: #1
  (void)&g<char>;

  // CHECK: declare {{.*}} @_ZN1SIA4_cE1gEv() #1
  (void)&S<char[4]>::g;
  // CHECK: declare {{.*}} @_ZN1SIcE1gEv()
  // CHECK-NOT: #1
  (void)&S<char>::g;
}

template<typename T> struct Nested {
  template<bool b, typename U> void f() noexcept(sizeof(T) == sizeof(U));
};

// CHECK: define {{.*}} @_Z1jv
void j() {
  // CHECK: declare {{.*}} @_ZN6NestedIiE1fILb1EcEEvv(
  // CHECK-NOT: #1
  Nested<int>().f<true, char>();
  // CHECK: declare {{.*}} @_ZN6NestedIlE1fILb0ElEEvv({{.*}}) #1
  Nested<long>().f<false, long>();
}

// CHECK: attributes #0 = { "target-features"={{.*}} }
// CHECK: attributes #1 = { nounwind "target-features"={{.*}} }
// CHECK: attributes #2 = { noinline noreturn nounwind }
