// RUN: %clang_cc1 -emit-llvm -o - -O1 %s | FileCheck %s
// Check that we generate TBAA for vtable pointer loads and stores.
struct A {
  virtual int foo() const ;
  virtual ~A();
};

void CreateA() {
  new A;
}

void CallFoo(A *a) {
  a->foo();
}

// CHECK: %{{.*}} = load {{.*}} !tbaa !0
// CHECK: store {{.*}} !tbaa !0
// CHECK: !0 = metadata !{metadata !"vtable pointer", metadata !1}
// CHECK: !1 = metadata !{metadata !"Simple C/C++ TBAA"}
