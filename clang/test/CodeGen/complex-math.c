// RUN: %clang_cc1 %s -O1 -emit-llvm -triple x86_64-unknown-unknown -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -O1 -emit-llvm -triple x86_64-pc-win64 -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -O1 -emit-llvm -triple i686-unknown-unknown -o - | FileCheck %s --check-prefix=X86

float _Complex add_float_rr(float a, float b) {
  // X86-LABEL: @add_float_rr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
float _Complex add_float_cr(float _Complex a, float b) {
  // X86-LABEL: @add_float_cr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
float _Complex add_float_rc(float a, float _Complex b) {
  // X86-LABEL: @add_float_rc(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
float _Complex add_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @add_float_cc(
  // X86: fadd
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}

float _Complex sub_float_rr(float a, float b) {
  // X86-LABEL: @sub_float_rr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
float _Complex sub_float_cr(float _Complex a, float b) {
  // X86-LABEL: @sub_float_cr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
float _Complex sub_float_rc(float a, float _Complex b) {
  // X86-LABEL: @sub_float_rc(
  // X86: fsub
  // X86: fsub float -0.{{0+}}e+00,
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
float _Complex sub_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @sub_float_cc(
  // X86: fsub
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}

float _Complex mul_float_rr(float a, float b) {
  // X86-LABEL: @mul_float_rr(
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
float _Complex mul_float_cr(float _Complex a, float b) {
  // X86-LABEL: @mul_float_cr(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
float _Complex mul_float_rc(float a, float _Complex b) {
  // X86-LABEL: @mul_float_rc(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
float _Complex mul_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @mul_float_cc(
  // X86-NOT: fmul
  // X86: call {{.*}} @__mulsc3(
  // X86: ret
  return a * b;
}

float _Complex div_float_rr(float a, float b) {
  // X86-LABEL: @div_float_rr(
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
float _Complex div_float_cr(float _Complex a, float b) {
  // X86-LABEL: @div_float_cr(
  // X86: fdiv
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
float _Complex div_float_rc(float a, float _Complex b) {
  // X86-LABEL: @div_float_rc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divsc3(
  // X86: ret
  return a / b;
}
float _Complex div_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @div_float_cc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divsc3(
  // X86: ret
  return a / b;
}

double _Complex add_double_rr(double a, double b) {
  // X86-LABEL: @add_double_rr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
double _Complex add_double_cr(double _Complex a, double b) {
  // X86-LABEL: @add_double_cr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
double _Complex add_double_rc(double a, double _Complex b) {
  // X86-LABEL: @add_double_rc(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
double _Complex add_double_cc(double _Complex a, double _Complex b) {
  // X86-LABEL: @add_double_cc(
  // X86: fadd
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}

double _Complex sub_double_rr(double a, double b) {
  // X86-LABEL: @sub_double_rr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
double _Complex sub_double_cr(double _Complex a, double b) {
  // X86-LABEL: @sub_double_cr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
double _Complex sub_double_rc(double a, double _Complex b) {
  // X86-LABEL: @sub_double_rc(
  // X86: fsub
  // X86: fsub double -0.{{0+}}e+00,
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
double _Complex sub_double_cc(double _Complex a, double _Complex b) {
  // X86-LABEL: @sub_double_cc(
  // X86: fsub
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}

double _Complex mul_double_rr(double a, double b) {
  // X86-LABEL: @mul_double_rr(
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
double _Complex mul_double_cr(double _Complex a, double b) {
  // X86-LABEL: @mul_double_cr(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
double _Complex mul_double_rc(double a, double _Complex b) {
  // X86-LABEL: @mul_double_rc(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
double _Complex mul_double_cc(double _Complex a, double _Complex b) {
  // X86-LABEL: @mul_double_cc(
  // X86-NOT: fmul
  // X86: call {{.*}} @__muldc3(
  // X86: ret
  return a * b;
}

double _Complex div_double_rr(double a, double b) {
  // X86-LABEL: @div_double_rr(
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
double _Complex div_double_cr(double _Complex a, double b) {
  // X86-LABEL: @div_double_cr(
  // X86: fdiv
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
double _Complex div_double_rc(double a, double _Complex b) {
  // X86-LABEL: @div_double_rc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divdc3(
  // X86: ret
  return a / b;
}
double _Complex div_double_cc(double _Complex a, double _Complex b) {
  // X86-LABEL: @div_double_cc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divdc3(
  // X86: ret
  return a / b;
}

long double _Complex add_long_double_rr(long double a, long double b) {
  // X86-LABEL: @add_long_double_rr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
long double _Complex add_long_double_cr(long double _Complex a, long double b) {
  // X86-LABEL: @add_long_double_cr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
long double _Complex add_long_double_rc(long double a, long double _Complex b) {
  // X86-LABEL: @add_long_double_rc(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
long double _Complex add_long_double_cc(long double _Complex a, long double _Complex b) {
  // X86-LABEL: @add_long_double_cc(
  // X86: fadd
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}

long double _Complex sub_long_double_rr(long double a, long double b) {
  // X86-LABEL: @sub_long_double_rr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
long double _Complex sub_long_double_cr(long double _Complex a, long double b) {
  // X86-LABEL: @sub_long_double_cr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
long double _Complex sub_long_double_rc(long double a, long double _Complex b) {
  // X86-LABEL: @sub_long_double_rc(
  // X86: fsub
  // X86: fsub x86_fp80 0xK8{{0+}},
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
long double _Complex sub_long_double_cc(long double _Complex a, long double _Complex b) {
  // X86-LABEL: @sub_long_double_cc(
  // X86: fsub
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}

long double _Complex mul_long_double_rr(long double a, long double b) {
  // X86-LABEL: @mul_long_double_rr(
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
long double _Complex mul_long_double_cr(long double _Complex a, long double b) {
  // X86-LABEL: @mul_long_double_cr(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
long double _Complex mul_long_double_rc(long double a, long double _Complex b) {
  // X86-LABEL: @mul_long_double_rc(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
long double _Complex mul_long_double_cc(long double _Complex a, long double _Complex b) {
  // X86-LABEL: @mul_long_double_cc(
  // X86-NOT: fmul
  // X86: call {{.*}} @__mulxc3(
  // X86: ret
  return a * b;
}

long double _Complex div_long_double_rr(long double a, long double b) {
  // X86-LABEL: @div_long_double_rr(
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
long double _Complex div_long_double_cr(long double _Complex a, long double b) {
  // X86-LABEL: @div_long_double_cr(
  // X86: fdiv
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
long double _Complex div_long_double_rc(long double a, long double _Complex b) {
  // X86-LABEL: @div_long_double_rc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divxc3(
  // X86: ret
  return a / b;
}
long double _Complex div_long_double_cc(long double _Complex a, long double _Complex b) {
  // X86-LABEL: @div_long_double_cc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divxc3(
  // X86: ret
  return a / b;
}
