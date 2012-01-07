// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodule-cache-path %t -I %S/Inputs %s -verify

@import namespaces_left;
@import namespaces_right;

void test() {
  int &ir1 = N1::f(1);
  int &ir2 = N2::f(1);
  int &ir3 = N3::f(1);
  float &fr1 = N1::f(1.0f);
  float &fr2 = N2::f(1.0f);
  double &dr1 = N2::f(1.0);
  double &dr2 = N3::f(1.0);
}
