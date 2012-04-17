// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -verify -fexceptions -fcxx-exceptions -triple x86_64-linux-gnu | FileCheck %s

void h();

template<typename T> void f() noexcept(sizeof(T) == 4) { h(); }

template<typename T> struct S {
  static void f() noexcept(sizeof(T) == 4) { h(); }
};

// CHECK: define {{.*}} @_Z1fIsEvv() {
template<> void f<short>() { h(); }
// CHECK: define {{.*}} @_Z1fIA2_sEvv() nounwind {
template<> void f<short[2]>() noexcept { h(); }

// CHECK: define {{.*}} @_ZN1SIsE1fEv()
// CHECK-NOT: nounwind
template<> void S<short>::f() { h(); }
// CHECK: define {{.*}} @_ZN1SIA2_sE1fEv() nounwind
template<> void S<short[2]>::f() noexcept { h(); }

// CHECK: define {{.*}} @_Z1fIDsEvv() {
template void f<char16_t>();
// CHECK: define {{.*}} @_Z1fIA2_DsEvv() nounwind {
template void f<char16_t[2]>();

// CHECK: define {{.*}} @_ZN1SIDsE1fEv()
// CHECK-NOT: nounwind
template void S<char16_t>::f();
// CHECK: define {{.*}} @_ZN1SIA2_DsE1fEv() nounwind
template void S<char16_t[2]>::f();

void g() {
  // CHECK: define {{.*}} @_Z1fIiEvv() nounwind {
  f<int>();
  // CHECK: define {{.*}} @_Z1fIA2_iEvv() {
  f<int[2]>();

  // CHECK: define {{.*}} @_ZN1SIiE1fEv() nounwind
  S<int>::f();
  // CHECK: define {{.*}} @_ZN1SIA2_iE1fEv()
  // CHECK-NOT: nounwind
  S<int[2]>::f();

  // CHECK: define {{.*}} @_Z1fIfEvv() nounwind {
  void (*f1)() = &f<float>;
  // CHECK: define {{.*}} @_Z1fIdEvv() {
  void (*f2)() = &f<double>;

  // CHECK: define {{.*}} @_ZN1SIfE1fEv() nounwind
  void (*f3)() = &S<float>::f;
  // CHECK: define {{.*}} @_ZN1SIdE1fEv()
  // CHECK-NOT: nounwind
  void (*f4)() = &S<double>::f;

  // CHECK: define {{.*}} @_Z1fIA4_cEvv() nounwind {
  (void)&f<char[4]>;
  // CHECK: define {{.*}} @_Z1fIcEvv() {
  (void)&f<char>;

  // CHECK: define {{.*}} @_ZN1SIA4_cE1fEv() nounwind
  (void)&S<char[4]>::f;
  // CHECK: define {{.*}} @_ZN1SIcE1fEv()
  // CHECK-NOT: nounwind
  (void)&S<char>::f;
}
