// RUN: %clang_cc1 -triple i386-unknown-windows-msvc -std=c++11 -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck %s

struct A {
  virtual ~A() {}
};

struct B {
  virtual ~B() {}
};

template<typename T>
struct AB: A, B {
};

template struct AB<int>;

// CHECK-LABEL: define {{.*}}@"\01??_E?$AB@H@@W3AEPAXI@Z"
// CHECK: call {{.*}}@"\01??_G?$AB@H@@UAEPAXI@Z"({{.*}}) #{{[0-9]*}}, !dbg [[THUNK_LOC:![0-9]*]]
// CHECK-LABEL: define

// CHECK: [[THUNK_VEC_DEL_DTOR:![0-9]*]] = distinct !DISubprogram({{.*}}function: {{.*}}@"\01??_E?$AB@H@@W3AEPAXI@Z"
// CHECK: [[THUNK_LOC]] = !DILocation(line: 15, scope: [[THUNK_VEC_DEL_DTOR]])
