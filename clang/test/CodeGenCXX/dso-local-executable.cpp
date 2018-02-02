// RUN: %clang_cc1 -mrelocation-model static -emit-llvm %s -o - | FileCheck --check-prefix=STATIC %s

// STATIC: @_ZTV1C = linkonce_odr dso_local unnamed_addr constant
// STATIC: @_ZTS1C = linkonce_odr dso_local constant
// STATIC: @_ZTI1C = linkonce_odr dso_local constant
// STATIC: @_ZZ14useStaticLocalvE3obj = linkonce_odr dso_local global
// STATIC: @_ZGVZN5guard1gEvE1a = linkonce_odr dso_local global
// STATIC: define dso_local void @_ZN1CC2Ev(
// STATIC: define dso_local void @_ZN1CC1Ev(
// STATIC: define linkonce_odr dso_local void @_ZN1C3fooEv(

struct C {
  C();
  virtual void foo() {}
};
C::C() {}

struct HasVTable {
  virtual void f();
};
inline HasVTable &useStaticLocal() {
  static HasVTable obj;
  return obj;
}
void useit() {
  useStaticLocal();
}

namespace guard {
int f();
inline int g() {
  static int a = f();
  return a;
}
int h() {
  return g();
}
} // namespace guard
