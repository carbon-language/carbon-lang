// RUN: %clang_cc1 -S -O3 -fno-builtin -o - %s | FileCheck %s
// rdar://10551066

double ceil(double x);
double copysign(double,double);
double cos(double x);
double fabs(double x);
double floor(double x);

double t1(double x) { return ceil(x); }
// CHECK: t1
// CHECK: ceil

double t2(double x, double y) { return copysign(x,y); }
// CHECK: t2
// CHECK: copysign

double t3(double x) { return cos(x); }
// CHECK: t3
// CHECK: cos

double t4(double x) { return fabs(x); }
// CHECK: t4
// CHECK: fabs

double t5(double x) { return floor(x); }
// CHECK: t5
// CHECK: floor
