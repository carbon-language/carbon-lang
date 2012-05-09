// RUN: %clang_cc1 -fbounds-checking=4 -emit-llvm -triple x86_64-apple-darwin10 < %s | FileCheck %s

// CHECK: @f
double f(int b, int i) {
  double a[b];
  return a[i];
  // CHECK: objectsize.i64({{.*}}, i1 false, i32 4)
  // CHECK: icmp uge i64 {{.*}}, 8
}

// CHECK: @f2
void f2() {
  int a[2];
  // CHECK: objectsize.i64({{.*}}, i1 false, i32 4)
  // CHECK: icmp uge i64 {{.*}}, 4
  a[1] = 42;
  
  short *b = malloc(64);
  // CHECK: objectsize.i64({{.*}}, i1 false, i32 4)
  // CHECK: icmp uge i64 {{.*}}, 4
  // CHECK: getelementptr {{.*}}, i64 5
  // CHECK: objectsize.i64({{.*}}, i1 false, i32 4)
  // CHECK: icmp uge i64 {{.*}}, 2
  b[5] = a[1]+2;
}
