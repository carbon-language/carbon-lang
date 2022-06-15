// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -triple x86_64-apple-darwin10 -I%S -emit-llvm -o - %s | FileCheck %s

#include <typeinfo>

// CHECK: @_ZTIDn = external constant i8*
int* a = nullptr;

void f() {
  int* a = nullptr;
}

typedef decltype(nullptr) nullptr_t;

nullptr_t get_nullptr();

struct X { };
void g() {
  // CHECK: call i8* @_Z11get_nullptrv()
  int (X::*pmf)(int) = get_nullptr();
}

const std::type_info& f2() {
  return typeid(nullptr_t);
}

union U {
  int n;
  nullptr_t b;
};
// CHECK-LABEL: define {{.*}}pr23833_a
// CHECK: store
// CHECK: load
// CHECK-NOT: load
// CHECK: ret i1 false
bool pr23833_a(U &u) { return bool(u.b); }

// CHECK-LABEL: define {{.*}}pr23833_b
// CHECK: store
// CHECK: load
// CHECK-NOT: load
// CHECK: ret i8* null
nullptr_t pr23833_b(nullptr_t &n) { return n; }

struct X1 { operator int*(); };
struct X2 { operator const nullptr_t&(); };

// CHECK-LABEL: define {{.*}}pr23833_c
// CHECK: call {{.*}}X1
// CHECK: call {{.*}}X2
// CHECK-NOT: load
// CHECK: ret i32
int pr23833_c() {
  return X1() != X2();
}

// CHECK-LABEL: define {{.*}}pr23833_d
// CHECK: call {{.*}}X2
// CHECK-NOT: load
// CHECK: store
// CHECK: load
// CHECK: ret i32*
int *pr23833_d() {
  int *p = X2();
  return p;
}

namespace PR39528 {
  constexpr nullptr_t null = nullptr;
  void f(nullptr_t);
  void g() { f(null); }
}
